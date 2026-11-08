from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.live_settings import LiveSettings
from config.rl_settings import RLSettings
from config.settings import ROOT_DIR
from src.core.events import DriftEvent, EventBus, KlineEvent, SentimentEvent
from src.data.buffer import MarketDataBuffer
from src.data.features import FeatureEngineer
from src.execution.paper import OrderExecutor, PaperGateway
from src.live.drift import ModelDriftMonitor, gate_exposure_on_critical
from src.risk import RiskManager
from src.risk.models import MarketSnapshot, PortfolioSource, PortfolioState
from src.rl.splits import FeatureScaler

log = logging.getLogger(__name__)


def find_model_dir(symbol: str, timeframe: str, root: Path | None = None) -> Path:
    root = root or ROOT_DIR / "data" / "models"
    tag = symbol.replace("/", "").split(":")[0]
    candidates = sorted(
        (
            path for path in root.glob(f"{tag}_{timeframe}_*")
            if (path / "scaler.json").is_file()
            and (path / "report.json").is_file()
            and (path / "best_model.zip").is_file()
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No hay un modelo completo para {symbol} ({timeframe}) en {root}. "
            "Entrena un modelo antes de iniciar main.py."
        )
    return candidates[-1]


class ModelBundle:
    def __init__(self, symbol: str, run_dir: Path, model: Any, scaler: FeatureScaler,
                 report: dict) -> None:
        self.symbol = symbol
        self.run_dir = run_dir
        self.model = model
        self.scaler = scaler
        self.report = report
        self.window = RLSettings.from_env().window
        self.n_features = len(scaler.columns)
        reference_path = run_dir / "drift_reference.npy"
        self.drift_reference = np.load(reference_path) if reference_path.is_file() else None

    @classmethod
    def load(cls, symbol: str, run_dir: Path) -> "ModelBundle":
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        algo = report.get("algo")
        from stable_baselines3 import A2C, PPO, SAC
        algorithm = {"a2c": A2C, "ppo": PPO, "sac": SAC}.get(algo)
        if algorithm is None:
            raise ValueError(f"Algoritmo de modelo no reconocido en {run_dir}: {algo!r}")
        model = algorithm.load(run_dir / "best_model")
        scaler = FeatureScaler.load(run_dir / "scaler.json")
        bundle = cls(symbol, run_dir, model, scaler, report)
        model_window = report.get("config", {}).get("window")
        if model_window is not None and int(model_window) != bundle.window:
            raise ValueError(
                f"RL_WINDOW ({bundle.window}) no coincide con el modelo ({model_window}) de {run_dir}."
            )
        return bundle


def target_weight_from_prediction(prediction, rl: RLSettings) -> float:
    action = float(np.clip(np.asarray(prediction, dtype=float).reshape(-1)[0], -1.0, 1.0))
    min_weight = -rl.max_leverage if rl.allow_short else 0.0
    return float(np.clip(action * rl.max_leverage, min_weight, rl.max_leverage))


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Estado inválido en {self.path}: se esperaba un objeto JSON.")
        return value

    def save(self, value: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


class TradeLog:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, filename: str, record: dict) -> None:
        path = self.state_dir / filename
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=str) + "\n")

    def equity(self, record: dict) -> None:
        self._write("equity.jsonl", record)

    def fill(self, record: dict) -> None:
        self._write("fills.jsonl", record)

    def closed_trade(self, record: dict) -> None:
        self._write("closed_trades.jsonl", record)

    def drift(self, record: dict) -> None:
        self._write("drift.jsonl", record)


def book_or_last_close(buffer: MarketDataBuffer, symbol: str) -> tuple[float, float]:
    book = buffer.order_book(symbol)
    candles = buffer.candles(symbol)
    if book is not None:
        volume = float(candles.iloc[-1]["volume"] * candles.iloc[-1]["close"]) if not candles.empty else 0.0
        return book.mid_price, volume
    if candles.empty:
        raise RuntimeError(f"No hay precio ni order book disponible para {symbol}.")
    last = candles.iloc[-1]
    return float(last["close"]), float(last["volume"] * last["close"])


def fresh_book(buffer: MarketDataBuffer, symbol: str) -> tuple[float, float]:
    return book_or_last_close(buffer, symbol)


class TradingOrchestrator:
    def __init__(
        self,
        symbols: list[str],
        timeframe: str,
        live: LiveSettings,
        rl: RLSettings,
        risk: RiskManager,
        buffer: MarketDataBuffer,
        engineer: FeatureEngineer,
        models: dict[str, ModelBundle],
        gateway: PaperGateway,
        executor: OrderExecutor,
        bus: EventBus,
        trade_log: TradeLog,
        store: StateStore,
        drift: dict[str, ModelDriftMonitor],
        backfill,
        mode: str,
        sentiment_enabled: bool = True,
        manual_approval: bool = False,
    ) -> None:
        self.symbols = symbols
        self.timeframe = timeframe
        self.live = live
        self.rl = rl
        self.risk = risk
        self.buffer = buffer
        self.engineer = engineer
        self.models = models
        self.gateway = gateway
        self.executor = executor
        self.bus = bus
        self.trade_log = trade_log
        self.store = store
        self.drift = drift
        self.backfill = backfill
        self.mode = mode
        self.sentiment_enabled = sentiment_enabled
        self.manual_approval = manual_approval
        self.book = gateway
        self.halt_file = live.state_dir / "HALT"
        self.started_at = datetime.now(timezone.utc)
        self._queue: asyncio.Queue[KlineEvent | None] = asyncio.Queue()
        self._stop = asyncio.Event()
        self._status_sym: dict[str, dict] = {}
        self._sentiment: dict[str, SentimentEvent] = {}
        self._pending_proposals: dict[str, dict[str, Any]] = {}
        self._proposal_context: dict[str, dict[str, Any]] = {}
        self._errors = 0
        self._stopped_at: str | None = None
        self._peak_equity = gateway.cash
        bus.subscribe(KlineEvent, self._on_kline)
        bus.subscribe(SentimentEvent, self._on_sentiment)

    def start(self) -> None:
        return None

    def _on_kline(self, event: KlineEvent) -> None:
        if event.is_closed and event.symbol in self.models:
            self._queue.put_nowait(event)

    def _on_sentiment(self, event: SentimentEvent) -> None:
        self._sentiment[event.asset] = event

    async def initialize(self) -> None:
        self._save()
        self._write_status()

    def request_stop(self) -> None:
        self._stopped_at = datetime.now(timezone.utc).isoformat()
        self._stop.set()
        self._queue.put_nowait(None)

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=30)
            except asyncio.TimeoutError:
                if self.halt_file.exists():
                    await self._halt()
                    break
                self._write_status()
                continue
            if event is None:
                break
            try:
                if self.halt_file.exists():
                    await self._halt()
                    break
                await self.process_bar(event)
                self._errors = 0
            except Exception:
                self._errors += 1
                log.exception("Error procesando vela de %s", event.symbol)
                self._write_status()
                if self._errors >= 5:
                    raise

    async def _halt(self) -> None:
        try:
            await self.flatten_all("halt_file")
        finally:
            self.request_stop()

    async def process_bar(self, event: KlineEvent) -> None:
        symbol = event.symbol
        candles = self.buffer.candles(symbol)
        features = self.engineer.transform(candles)
        bundle = self.models[symbol]
        if len(features) < max(bundle.window, self.engineer.warmup_periods):
            return
        row = features.iloc[-1]
        scaled = bundle.scaler.transform(features.tail(bundle.window))
        price, quote_volume = book_or_last_close(self.buffer, symbol)
        qty = self.gateway.qty(symbol)
        equity = self._equity()
        if equity <= 0:
            raise RuntimeError("Equity no positivo; se detiene paper trading.")
        current_weight = qty * price / equity
        entry = self.gateway.entry_price(symbol)
        drawdown = max(0.0, 1 - equity / max(self._peak_equity, equity))
        scaled_features = bundle.scaler.transform(features.tail(1))[0]
        drift_status = self.drift[symbol].evaluate(scaled_features, drawdown)
        unrealized = math.copysign(1.0, qty) * (price / entry - 1) if qty and entry else 0.0
        observation = np.concatenate([
            scaled.reshape(-1),
            np.asarray([current_weight, np.clip(unrealized / 0.05, -5, 5), np.clip(drawdown / 0.05, 0, 5)]),
        ]).astype(np.float32)
        prediction, _ = bundle.model.predict(observation, deterministic=True)
        requested = target_weight_from_prediction(prediction, self.rl)
        requested_for_risk = requested
        if drift_status.entries_paused:
            requested_for_risk = gate_exposure_on_critical(current_weight, requested)
        returns = features["close"].pct_change().dropna().to_numpy(dtype=float)
        other_exposure = sum(
            abs(self.gateway.qty(other) * book_or_last_close(self.buffer, other)[0] / equity)
            for other in self.symbols if other != symbol
        )
        sentiment = self._sentiment.get(symbol.split("/")[0])
        state = PortfolioState(
            symbol=symbol,
            equity=equity,
            weight=current_weight,
            price=price,
            timestamp=pd.Timestamp(event.open_time, unit="ms", tz="UTC").to_pydatetime(),
            entry_price=entry,
            other_exposure=other_exposure,
            source=PortfolioSource.SIMULATION,
        )
        market = MarketSnapshot(
            price=price,
            atr_pct=float(row["atr_pct"]),
            high=float(event.high),
            low=float(event.low),
            spread_bps=self.buffer.order_book(symbol).spread_bps if self.buffer.order_book(symbol) else None,
            bar_quote_volume=quote_volume,
        )
        decision = await self.risk.evaluate_and_publish(state, market, requested_for_risk, returns, sentiment)
        before_qty = qty
        before_entry = self.gateway.entry_price(symbol)
        proposal = None
        if self.manual_approval:
            self._discard_symbol_proposal(symbol)
            increases_risk = (
                decision.approved_weight * current_weight >= 0
                and abs(decision.approved_weight) > abs(current_weight)
            )
            below_rebalance_minimum = (
                increases_risk
                and abs(decision.approved_weight - current_weight) < self.executor.settings.min_rebalance
            )
            if (
                abs(decision.approved_weight - current_weight) > 1e-9
                and not below_rebalance_minimum
            ):
                proposal = self._create_proposal(
                    event=event,
                    price=price,
                    equity=equity,
                    current_weight=current_weight,
                    requested=requested,
                    decision=decision,
                )
                self._proposal_context[proposal["id"]] = {
                    "event": event,
                    "returns": returns.copy(),
                    "requested": requested,
                    "entries_paused": drift_status.entries_paused,
                    "sentiment": sentiment,
                    "atr_pct": float(row["atr_pct"]),
                }
            execution = None
        else:
            execution = await self.executor.execute(
                symbol, decision.approved_weight, equity, price, quote_volume
            )
        if execution:
            self.trade_log.fill({**execution, "bar": event.open_time, "mode": self.mode})
            await self._record_position_change(
                symbol, before_qty, before_entry, execution, float(row["atr_pct"]), event
            )
        if drift_status.level in {"WARN", "CRITICAL"}:
            bar = pd.Timestamp(event.open_time, unit="ms", tz="UTC").isoformat()
            self.trade_log.drift({
                "bar": bar, "symbol": symbol, **drift_status.to_dict(),
            })
            await self.bus.publish(DriftEvent(
                bar=bar,
                symbol=symbol,
                level=drift_status.level,
                n_drifted=drift_status.n_drifted,
                n_features=drift_status.n_features,
                drifted=drift_status.drifted,
                reasons=drift_status.reasons,
                tstat=drift_status.psi_max,
                drawdown=drift_status.drawdown,
            ))
        self._peak_equity = max(self._peak_equity, self._equity())
        self.trade_log.equity({
            "bar": pd.Timestamp(event.open_time, unit="ms", tz="UTC").isoformat(),
            "symbol": symbol, "equity": self._equity(), "price": price,
            "weight": self.gateway.qty(symbol) * price / max(self._equity(), 1e-12),
            "requested": requested, "approved": decision.approved_weight,
            "risk_requested": decision.requested_weight,
            "action": decision.action.value, "codes": list(decision.codes),
            "breaker": decision.breaker.value,
        })
        self._status_sym[symbol] = {
            "bar": pd.Timestamp(event.open_time, unit="ms", tz="UTC").isoformat(),
            "price": price, "qty": self.gateway.qty(symbol),
            "weight": self.gateway.qty(symbol) * price / max(self._equity(), 1e-12),
            "requested": requested, "risk_requested": decision.requested_weight,
            "approved": decision.approved_weight,
            "action": decision.action.value, "codes": list(decision.codes),
            "reasons": [r.to_dict() if hasattr(r, "to_dict") else {
                "code": r.code, "message": r.message, "before": r.before, "after": r.after
            } for r in decision.reasons],
            "entry": self.gateway.entry_price(symbol),
            "stop": decision.stop_price, "take_profit": decision.take_profit_price,
            "breaker": decision.breaker.value, "protection": "Stop gestionado localmente (paper)",
            "last_exec": execution, "drift": drift_status.to_dict(), "model": {
                "run": bundle.run_dir.name, "n_features": bundle.n_features,
                "window": bundle.window, "val": bundle.report.get("val", {}),
            },
            "agent_action": requested, "spread_bps": market.spread_bps,
            "atr_pct": market.atr_pct,
            "kelly_cap": decision.kelly_cap, "var_cap": decision.var_cap,
            "equity": self._equity(), "entries_paused": drift_status.entries_paused, "stale": False,
            "pending_proposal": proposal,
        }
        self._save()
        self._write_status()

    def _discard_symbol_proposal(self, symbol: str) -> None:
        old = self._pending_proposals.pop(symbol, None)
        if old is not None:
            self._proposal_context.pop(old["id"], None)
            status = self._status_sym.get(symbol)
            if status is not None:
                status["pending_proposal"] = None

    def _create_proposal(
        self,
        event: KlineEvent,
        price: float,
        equity: float,
        current_weight: float,
        requested: float,
        decision,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        delta_weight = decision.approved_weight - current_weight
        quantity = delta_weight * equity / price
        proposal = {
            "id": uuid.uuid4().hex,
            "symbol": event.symbol,
            "bar": pd.Timestamp(event.open_time, unit="ms", tz="UTC").isoformat(),
            "created_at": now.isoformat(),
            "expires_at": (now + pd.Timedelta(seconds=120)).isoformat(),
            "price": price,
            "equity": equity,
            "current_weight": current_weight,
            "requested_weight": requested,
            "approved_weight": decision.approved_weight,
            "side": "BUY" if quantity > 0 else "SELL",
            "quantity": abs(quantity),
            "notional": abs(quantity * price),
            "risk_action": decision.action.value,
            "risk_codes": list(decision.codes),
            "risk_reasons": [
                r.to_dict() if hasattr(r, "to_dict") else {
                    "code": r.code, "message": r.message, "before": r.before, "after": r.after,
                }
                for r in decision.reasons
            ],
            "mode": self.mode,
        }
        self._pending_proposals[event.symbol] = proposal
        return proposal

    async def handle_manual_approval(self, proposal_id: str, approve: bool) -> dict[str, Any]:
        proposal = next(
            (p for p in self._pending_proposals.values() if p["id"] == proposal_id),
            None,
        )
        context = self._proposal_context.get(proposal_id)
        if proposal is None or context is None:
            raise ValueError("La propuesta ya no está disponible; espera la siguiente señal.")

        symbol = proposal["symbol"]
        event = context["event"]
        candles = self.buffer.candles(symbol)
        expected_bar = pd.Timestamp(event.open_time, unit="ms", tz="UTC")
        if candles.empty or candles.index[-1] != expected_bar:
            self._discard_symbol_proposal(symbol)
            self._write_status()
            raise ValueError("La propuesta caducó porque llegó una vela nueva.")
        if datetime.now(timezone.utc) >= datetime.fromisoformat(proposal["expires_at"]):
            self._discard_symbol_proposal(symbol)
            self._write_status()
            raise ValueError("La propuesta caducó; espera la siguiente señal.")

        self._discard_symbol_proposal(symbol)
        if not approve:
            status = self._status_sym.get(symbol)
            if status is not None:
                status["approval_result"] = "REJECTED"
                status["pending_proposal"] = None
            self._write_status()
            return {"status": "rejected", "symbol": symbol}

        price, quote_volume = book_or_last_close(self.buffer, symbol)
        equity = self._equity()
        if equity <= 0:
            raise RuntimeError("Equity no positivo; se cancela la propuesta.")
        current_qty = self.gateway.qty(symbol)
        current_weight = current_qty * price / equity
        requested_for_risk = context["requested"]
        if context["entries_paused"]:
            requested_for_risk = gate_exposure_on_critical(current_weight, requested_for_risk)
        book = self.buffer.order_book(symbol)
        state = PortfolioState(
            symbol=symbol,
            equity=equity,
            weight=current_weight,
            price=price,
            timestamp=datetime.now(timezone.utc),
            entry_price=self.gateway.entry_price(symbol),
            other_exposure=sum(
                abs(self.gateway.qty(other) * book_or_last_close(self.buffer, other)[0] / equity)
                for other in self.symbols if other != symbol
            ),
            source=PortfolioSource.SIMULATION,
        )
        market = MarketSnapshot(
            price=price,
            atr_pct=context["atr_pct"],
            high=float(event.high),
            low=float(event.low),
            spread_bps=book.spread_bps if book else None,
            bar_quote_volume=quote_volume,
        )
        decision = await self.risk.evaluate_and_publish(
            state, market, requested_for_risk, context["returns"], context["sentiment"]
        )
        if not math.isclose(
            decision.approved_weight, proposal["approved_weight"], rel_tol=0.0, abs_tol=1e-6
        ):
            self._write_status()
            raise ValueError(
                "El riesgo cambió desde la propuesta; no se ejecutó. Espera una señal actualizada."
            )

        before_qty = current_qty
        before_entry = self.gateway.entry_price(symbol)
        execution = await self.executor.execute(
            symbol, decision.approved_weight, equity, price, quote_volume
        )
        if execution:
            self.trade_log.fill({**execution, "bar": event.open_time, "mode": self.mode})
            await self._record_position_change(
                symbol, before_qty, before_entry, execution, context["atr_pct"], event
            )
        else:
            raise ValueError("El ejecutor no generó una orden para esta propuesta.")
        status = self._status_sym.get(symbol)
        if status is not None:
            status.update({
                "price": price,
                "qty": self.gateway.qty(symbol),
                "weight": self.gateway.qty(symbol) * price / max(self._equity(), 1e-12),
                "approved": decision.approved_weight,
                "action": decision.action.value,
                "codes": list(decision.codes),
                "entry": self.gateway.entry_price(symbol),
                "last_exec": execution,
                "equity": self._equity(),
                "pending_proposal": None,
                "approval_result": "APPROVED_AND_EXECUTED",
            })
        self._save()
        self._write_status()
        return {"status": "executed", "symbol": symbol, "execution": execution}

    async def _record_position_change(self, symbol: str, old_qty: float, old_entry: float | None,
                                      execution: dict, atr_pct: float, event: KlineEvent) -> None:
        new_qty = self.gateway.qty(symbol)
        price = float(execution["price"])
        if abs(old_qty) <= 1e-12 and abs(new_qty) > 1e-12:
            self.risk.on_position_opened(symbol, 1 if new_qty > 0 else -1, price, atr_pct)
        elif abs(old_qty) > 1e-12 and abs(new_qty) <= 1e-12 and old_entry:
            self._record_closed_trade(symbol, old_qty, old_entry, price, execution, event)
        elif old_qty * new_qty < 0 and old_entry:
            self._record_closed_trade(symbol, old_qty, old_entry, price, execution, event)
            self.risk.on_position_opened(symbol, 1 if new_qty > 0 else -1, price, atr_pct)

    def _record_closed_trade(self, symbol: str, old_qty: float, old_entry: float,
                             price: float, execution: dict, event: KlineEvent) -> None:
        pnl = self.risk.on_position_closed(
            symbol, 1 if old_qty > 0 else -1, old_entry, price,
            pd.Timestamp(event.open_time, unit="ms", tz="UTC").to_pydatetime(),
            fees_pct=float(execution["fee"]) / max(float(execution["notional"]), 1e-12),
        )
        self.trade_log.closed_trade({
            "symbol": symbol, "exit_price": price, "pnl": pnl,
            "bar": pd.Timestamp(event.open_time, unit="ms", tz="UTC").isoformat(),
        })

    def _equity(self) -> float:
        prices = {symbol: book_or_last_close(self.buffer, symbol)[0] for symbol in self.symbols
                  if not self.buffer.candles(symbol).empty}
        return self.gateway.equity(prices)

    def _last_price(self, symbol: str) -> float:
        return book_or_last_close(self.buffer, symbol)[0]

    async def _account(self, symbol: str, price: float) -> tuple[float, float]:
        equity = self._equity()
        weight = self.gateway.qty(symbol) * price / equity if equity > 0 else 0.0
        return equity, weight

    async def flatten_all(self, reason: str) -> None:
        for symbol in self.symbols:
            qty = self.gateway.qty(symbol)
            if qty == 0:
                continue
            price, quote_volume = book_or_last_close(self.buffer, symbol)
            equity = self._equity()
            await self.executor.execute(symbol, 0.0, equity, price, quote_volume)
            log.warning("Paper posición cerrada %s por %s", symbol, reason)

    def _save(self) -> None:
        self.store.save({
            "mode": self.mode,
            "cash": self.gateway.cash,
            "positions": {
                symbol: {"qty": self.gateway.qty(symbol), "entry_price": self.gateway.entry_price(symbol)}
                for symbol in self.symbols
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def _write_status(self) -> None:
        try:
            self.store_status.save({
                "mode": self.mode,
                "started_at": self.started_at.isoformat(),
                "stopped_at": self._stopped_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "manual_approval": self.manual_approval,
                "pending_proposals": list(self._pending_proposals.values()),
                "symbols": self._status_sym,
                "sentiment": self._sentiment_snapshot(),
                "open_positions": {
                    symbol: {"qty": self.gateway.qty(symbol), "entry_price": self.gateway.entry_price(symbol)}
                    for symbol in self.symbols if abs(self.gateway.qty(symbol)) > 1e-12
                },
                "equity": self._equity(),
                "errors": self._errors,
                "queue": self._queue.qsize(),
                "pid": __import__("os").getpid(),
                "tf_seconds": pd.Timedelta(self.timeframe).total_seconds()
                    if self.timeframe.endswith(("min", "h", "d")) else 3600,
            })
        except Exception:
            log.exception("No se pudo actualizar status.json")

    @property
    def store_status(self) -> StateStore:
        return StateStore(self.live.state_dir / "status.json")

    def _sentiment_snapshot(self) -> dict[str, dict]:
        now = datetime.now(timezone.utc)
        assets = sorted({symbol.split("/")[0] for symbol in self.symbols})
        snapshot: dict[str, dict] = {}
        for asset in assets:
            event = self._sentiment.get(asset)
            if event is None:
                snapshot[asset] = {
                    "status": "PENDING" if self.sentiment_enabled else "DISABLED",
                    "reason": "esperando el primer resultado del servicio" if self.sentiment_enabled
                    else "deshabilitado con --no-sentiment",
                    "score": None,
                    "confidence": 0.0,
                    "n_scored": 0,
                    "degraded": True,
                    "age_min": None,
                }
                continue
            timestamp = datetime.fromtimestamp(event.timestamp / 1000, tz=timezone.utc)
            snapshot[asset] = {
                "status": "DEGRADED" if event.degraded else "ACTIVE",
                "score": event.score,
                "confidence": event.confidence,
                "n_scored": event.n_scored,
                "fear_greed": event.fear_greed,
                "drivers": list(event.drivers),
                "degraded": event.degraded,
                "timestamp": timestamp.isoformat(),
                "age_min": max(0.0, (now - timestamp).total_seconds() / 60),
            }
        return snapshot
