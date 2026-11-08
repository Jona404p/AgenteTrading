from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from config.risk_settings import RiskSettings
from src.core.events import EventBus, RiskEvent
from src.risk.audit import RiskAuditLog
from src.risk.breaker import CircuitBreaker
from src.risk.guardrails import (
    HARD_MAX_LEVERAGE,
    HARD_MAX_ORDER_EQUITY_FRACTION,
    HARD_MAX_STOP_ATR_MULT,
)
from src.risk.models import (BreakerState, MarketSnapshot, PortfolioState, RiskAction,
                             PortfolioSource, RiskDecision, RiskReason, SentimentLike)
from src.risk.sizing import KellySizer, TradeStats
from src.risk.stops import PositionStops
from src.risk.var import VaRModel, max_weight_for_var

log = logging.getLogger(__name__)

_EPS = 1e-9
CVAR_LIMIT_MULT = 1.5      # el expected shortfall puede ser hasta 1.5× el límite de VaR


def _sgn(x: float) -> int:
    return 0 if abs(x) < _EPS else (1 if x > 0 else -1)


def _no_increase(w: float, cur: float) -> float:
    """Mayor exposición ≤ w que no aumenta el riesgo respecto a `cur` (no abre ni invierte)."""
    if _sgn(cur) == 0 or _sgn(w) != _sgn(cur):
        return 0.0
    return math.copysign(min(abs(w), abs(cur)), cur)


class RiskManager:
    """Filtro de riesgo entre el agente RL (exposición objetivo) y la ejecución.

    Reglas DURAS (pueden forzar reducir una posición abierta): stops SL/TP/trailing, circuit breaker,
    shorts no permitidos, exposición/apalancamiento máximos, límites de VaR/CVaR.
    Reglas de ENTRADA (solo impiden AUMENTAR riesgo): Kelly, cooldown tras stop, spread, participación,
    y el overlay de sentimiento (solo recorta el objetivo; nunca añade exposición).
    Todo lo que reduce riesgo (cerrar/recortar) se permite siempre, incluso con el breaker activo.

    Uso en F5:  decision = rm.evaluate(state, market, agent_weight, returns, sentiment)
    ejecutar `decision.approved_weight`; tras cada fill llamar a on_position_opened / on_position_closed.
    """

    def __init__(self, cfg: RiskSettings, bus: EventBus | None = None,
                 persist: bool = True, audit: RiskAuditLog | None = None,
                 require_exchange_state: bool = True) -> None:
        """Reject simulated portfolio snapshots by default; opt out only in simulations/tests."""
        self.cfg, self.bus, self.persist = cfg, bus, persist
        self.require_exchange_state = require_exchange_state
        self.breaker = CircuitBreaker(cfg, cfg.state_dir / "breaker.json" if persist else None)
        self.stats = TradeStats(cfg.kelly_window_trades)
        self._trades_path = cfg.state_dir / "trades.json"
        if persist and self._trades_path.exists():
            for p in json.loads(self._trades_path.read_text()):
                self.stats.add(p)
        self.audit = audit or (RiskAuditLog(cfg.state_dir / "decisions.jsonl") if persist else None)
        self.kelly = KellySizer(cfg.kelly_fraction, cfg.kelly_min_trades, cfg.kelly_bootstrap_cap,
                                cfg.kelly_floor, min(cfg.max_position_frac, cfg.max_leverage))
        self.var = VaRModel(cfg.var_confidence, cfg.var_horizon_bars, cfg.var_window, cfg.var_min_obs)
        self._stops: dict[str, PositionStops] = {}
        self._reentry_until: dict[str, datetime] = {}

    def _new_stops(self, side: int, entry_price: float, atr_pct: float) -> PositionStops:
        configured_sl = self.cfg.sl_atr_mult
        sl_mult = (
            min(configured_sl, HARD_MAX_STOP_ATR_MULT)
            if math.isfinite(configured_sl) and configured_sl > 0
            else HARD_MAX_STOP_ATR_MULT
        )
        return PositionStops(
            side, entry_price, atr_pct, sl_mult, self.cfg.tp_atr_mult, self.cfg.trail_atr_mult
        )

    # ------------------------------------------------------------ eventos de ejecución
    def on_position_opened(self, symbol: str, side: int, entry_price: float, atr_pct: float) -> None:
        self._stops[symbol] = self._new_stops(side, entry_price, atr_pct)

    def current_stops(self) -> dict[str, dict[str, float | int | None]]:
        """Return a detached, read-only-by-convention snapshot of active protective levels."""
        return {
            symbol: {
                "side": tracker.side,
                "entry": tracker.entry,
                "stop": tracker.stop,
                "initial_stop": tracker.initial_stop,
                "take_profit": tracker.take_profit,
                "extreme": tracker.extreme,
            }
            for symbol, tracker in self._stops.items()
        }

    def on_position_closed(self, symbol: str, side: int, entry_price: float, exit_price: float,
                           now: datetime, reason: str | None = None, fees_pct: float = 0.0) -> float:
        """Registra una operación cerrada: alimenta Kelly y el contador de pérdidas. Devuelve el PnL neto (fracción)."""
        pnl = side * (exit_price / entry_price - 1.0) - fees_pct
        self.stats.add(pnl)
        self._stops.pop(symbol, None)
        if reason in ("stop_loss", "trailing_stop") and self.cfg.reentry_cooldown_minutes > 0:
            self._reentry_until[symbol] = now + timedelta(minutes=self.cfg.reentry_cooldown_minutes)
        self.breaker.register_trade(self.stats.consecutive_losses, now)
        if self.persist:
            self._trades_path.write_text(json.dumps(self.stats.to_list()))
        return pnl

    # ------------------------------------------------------------ decisión
    def evaluate(self, state: PortfolioState, market: MarketSnapshot, requested_weight: float,
                 returns, sentiment: SentimentLike | None = None) -> RiskDecision:
        if self.require_exchange_state and state.source is not PortfolioSource.EXCHANGE:
            raise ValueError(
                "RiskManager requiere PortfolioState conciliado con el exchange; "
                "usa require_exchange_state=False solo para simulación/tests."
            )
        c = self.cfg
        now, sym, cur = state.timestamp, state.symbol, state.weight
        req = float(requested_weight)
        numeric_inputs = (
            state.equity, state.weight, state.price, state.other_exposure, req,
            market.price, market.atr_pct,
            *(() if market.high is None else (market.high,)),
            *(() if market.low is None else (market.low,)),
            *(() if market.spread_bps is None else (market.spread_bps,)),
            *(() if market.bar_quote_volume is None else (market.bar_quote_volume,)),
        )
        if any(not math.isfinite(value) for value in numeric_inputs):
            raise ValueError("La decisión de riesgo requiere entradas numéricas finitas.")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("El timestamp de la decisión de riesgo debe incluir zona horaria.")
        if (state.price <= 0 or state.other_exposure < 0 or market.price <= 0
                or market.atr_pct < 0
                or (market.high is not None and market.high <= 0)
                or (market.low is not None and market.low <= 0)
                or (market.spread_bps is not None and market.spread_bps < 0)
                or (market.bar_quote_volume is not None and market.bar_quote_volume < 0)):
            raise ValueError("La decisión de riesgo recibió valores de mercado o cartera inválidos.")
        w = req
        reasons: list[RiskReason] = []
        flatten = False

        def set_w(new: float, code: str, msg: str) -> None:
            nonlocal w
            if abs(new - w) > _EPS:
                reasons.append(RiskReason(code, msg, w, new))
                w = new

        # 0) Circuit breaker
        breaker = self.breaker.update(state.equity, now)
        if state.equity <= 0:
            flatten = _sgn(cur) != 0
            set_w(0.0, "NON_POSITIVE_EQUITY", "equity agotado: no se permite mantener exposición")

        # 1) Stops de la posición abierta
        tracker = self._stops.get(sym)
        if _sgn(cur) == 0:
            self._stops.pop(sym, None)
            tracker = None
        elif tracker is None:    # recuperación tras reinicio: reconstruye con los datos disponibles
            tracker = self._new_stops(_sgn(cur), state.entry_price or state.price, market.atr_pct)
            self._stops[sym] = tracker
        if tracker is not None:
            trig = tracker.update(market.high if market.high is not None else market.price,
                                  market.low if market.low is not None else market.price)
            if trig is not None:
                flatten = True
                if c.reentry_cooldown_minutes > 0 and trig.kind != "take_profit":
                    self._reentry_until[sym] = now + timedelta(minutes=c.reentry_cooldown_minutes)
                set_w(0.0, trig.kind.upper(), f"{trig.kind} tocado en {trig.price:.4f}")

        # 2) Circuit breaker → recortes
        reduce_only = breaker is not BreakerState.OK
        if breaker is BreakerState.DRAWDOWN_HALT and c.flatten_on_drawdown_halt and _sgn(cur) != 0:
            flatten = True
            set_w(0.0, "BREAKER_DRAWDOWN", "drawdown máximo superado: cerrar y detener")
        elif breaker is BreakerState.DAILY_HALT and _sgn(cur) != 0:
            flatten = True
            set_w(0.0, "BREAKER_DAILY", "pérdida diaria máxima superada: liquidar y detener")
        elif reduce_only:
            set_w(_no_increase(w, cur), f"BREAKER_{breaker.value}", "circuit breaker activo: solo reducir")

        # 3) Tope inquebrantable al tamaño de cada cambio de exposición.
        if not flatten and abs(w - cur) > HARD_MAX_ORDER_EQUITY_FRACTION + _EPS:
            capped = cur + math.copysign(HARD_MAX_ORDER_EQUITY_FRACTION, w - cur)
            set_w(capped, "HARD_ORDER_LIMIT",
                  f"orden limitada a {HARD_MAX_ORDER_EQUITY_FRACTION:.0%} del equity")

        # 4) Límites duros de exposición y apalancamiento
        if w < 0 and not c.allow_short:
            set_w(0.0, "NO_SHORT", "cortos no permitidos en este mercado")
        configured_caps = (c.max_position_frac, c.max_leverage, c.max_total_exposure)
        if any(not math.isfinite(value) or value < 0 for value in configured_caps):
            cap = 0.0
            flatten = _sgn(cur) != 0
            set_w(0.0, "INVALID_RISK_LIMIT", "límite de riesgo inválido: se bloquea la exposición")
        else:
            headroom = max(min(c.max_total_exposure, HARD_MAX_LEVERAGE) - state.other_exposure, 0.0)
            cap = min(c.max_position_frac, c.max_leverage, HARD_MAX_LEVERAGE, headroom)
        if abs(w) > cap + _EPS:
            set_w(math.copysign(cap, w), "MAX_EXPOSURE", f"tope de exposición {cap:.2f}")

        # 5) VaR / CVaR (límite duro)
        var_cap = var_unit = None
        if _sgn(w) != 0:
            v = self.var.compute(returns, _sgn(w), market.atr_pct)
            var_unit = v.var
            cap_v = min(max_weight_for_var(c.var_limit_pct, v.var),
                        max_weight_for_var(c.var_limit_pct * CVAR_LIMIT_MULT, v.cvar))
            var_cap = cap_v if math.isfinite(cap_v) else None
            if abs(w) > cap_v + _EPS:
                set_w(math.copysign(cap_v, w), "VAR_LIMIT",
                      f"VaR{c.var_confidence:.0%}/{c.var_horizon_bars}b={v.var:.2%}/u ({v.method}) → máx {cap_v:.2f}")

        # 6) Overlay de sentimiento: solo recorta el objetivo
        if (sentiment is not None and not sentiment.degraded
                and sentiment.confidence >= c.sentiment_min_confidence):
            if (w > 0 and sentiment.score <= c.sentiment_bearish) or (w < 0 and sentiment.score >= c.sentiment_bullish):
                set_w(w * c.sentiment_derisk_factor, "SENTIMENT",
                      f"sentimiento {sentiment.score:.0f} (conf {sentiment.confidence:.2f}) contrario a la posición")

        # 7) Reglas de entrada: solo actúan si el objetivo AUMENTA el riesgo
        increases = (_sgn(w) != 0) and (_sgn(w) != _sgn(cur) or abs(w) > abs(cur) + _EPS)
        kelly_cap, kelly_mode = self.kelly.cap(self.stats)
        if increases:
            allowed = max(kelly_cap, abs(cur) if _sgn(w) == _sgn(cur) else 0.0)
            if abs(w) > allowed + _EPS:
                set_w(math.copysign(allowed, w), "KELLY_CAP", f"Kelly fraccional ({kelly_mode}) → máx {kelly_cap:.2f}")
            if now < self._reentry_until.get(sym, datetime.min.replace(tzinfo=timezone.utc)):
                set_w(_no_increase(w, cur), "REENTRY_COOLDOWN", "espera tras stop-loss")
            if market.spread_bps is not None and market.spread_bps > c.max_spread_bps:
                set_w(_no_increase(w, cur), "SPREAD", f"spread {market.spread_bps:.1f} bps > {c.max_spread_bps:.0f}")
            if market.bar_quote_volume and _sgn(w) != 0 and abs(w) > abs(cur) + _EPS:
                max_delta = c.max_participation * market.bar_quote_volume / max(state.equity, _EPS)
                if abs(w - cur) > max_delta:
                    set_w(cur + math.copysign(max_delta, w - cur), "PARTICIPATION",
                          f"orden > {c.max_participation:.1%} del volumen de la vela")
        if reduce_only:                       # el breaker manda sobre cualquier ajuste posterior
            set_w(_no_increase(w, cur), f"BREAKER_{breaker.value}", "circuito activo: sin aumentos")

        # 8) El mínimo notional solo afecta entradas; nunca impide una reducción forzada.
        delta_notional = abs(w - cur) * state.equity
        increases = (_sgn(w) != 0) and (_sgn(w) != _sgn(cur) or abs(w) > abs(cur) + _EPS)
        if not flatten and increases and _EPS < delta_notional < c.min_notional:
            set_w(cur, "MIN_NOTIONAL", f"orden {delta_notional:.2f} < mínimo {c.min_notional:.0f}")

        # 9) Clasificación
        if flatten:
            action = RiskAction.FLATTEN
        elif abs(w - req) <= _EPS:
            action = RiskAction.ALLOW
        elif any(r.code == "MIN_NOTIONAL" for r in reasons):
            action = RiskAction.HOLD
        elif abs(w - cur) <= _EPS:
            action = RiskAction.BLOCK
        else:
            action = RiskAction.REDUCE

        stop_price = tp_price = None
        t = self._stops.get(sym)
        if t is not None:
            stop_price, tp_price = t.stop, t.take_profit
        elif _sgn(w) != 0:       # vista previa para una posición que se abriría ahora
            pv = self._new_stops(_sgn(w), state.price, market.atr_pct)
            stop_price, tp_price = pv.stop, pv.take_profit

        decision = RiskDecision(
            symbol=sym, timestamp=now, requested_weight=req, approved_weight=w, current_weight=cur,
            action=action, reasons=tuple(reasons), stop_price=stop_price, take_profit_price=tp_price,
            breaker=breaker, kelly_cap=kelly_cap, var_cap=var_cap, var_unit=var_unit)
        if self.audit and (reasons or action is not RiskAction.ALLOW):
            self.audit.write(decision)
        if reasons:
            log.info("[riesgo] %s %s %.2f→%.2f (%s)", sym, action.value, req, w, ",".join(decision.codes))
        return decision

    async def evaluate_and_publish(self, *args, **kwargs) -> RiskDecision:
        d = self.evaluate(*args, **kwargs)
        if self.bus is not None:
            await self.bus.publish(RiskEvent(
                symbol=d.symbol, action=d.action.value, requested_weight=d.requested_weight,
                approved_weight=d.approved_weight, reasons=d.codes, breaker=d.breaker.value,
                timestamp=int(d.timestamp.timestamp() * 1000)))
        return d
