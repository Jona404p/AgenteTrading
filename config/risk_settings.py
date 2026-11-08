from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from config.settings import ROOT_DIR, MarketType, Settings


def _f(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _b(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


_TF_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def bars_per_day(timeframe: str) -> float:
    return 86400 / (int(timeframe[:-1]) * _TF_SECONDS[timeframe[-1]])


@dataclass(frozen=True)
class RiskSettings:
    # ── Tamaño de posición (Kelly) ──
    kelly_fraction: float          # multiplicador del Kelly completo (0.25 = cuarto de Kelly)
    kelly_min_trades: int          # operaciones cerradas necesarias para confiar en el Kelly empírico
    kelly_window_trades: int       # ventana móvil de operaciones
    kelly_bootstrap_cap: float     # tope de exposición mientras no hay suficientes operaciones
    kelly_floor: float             # exposición mínima permitida aun con Kelly ≤ 0 (permite recuperar estadística)
    # ── Límites de exposición ──
    max_position_frac: float       # |exposición| máxima por símbolo (fracción del equity)
    max_total_exposure: float      # suma de |exposición| de todos los símbolos
    max_leverage: float
    allow_short: bool
    # ── VaR ──
    var_confidence: float
    var_horizon_bars: int
    var_limit_pct: float           # pérdida VaR máxima tolerada (fracción del equity) al horizonte
    var_window: int
    var_min_obs: int
    # ── Circuit breaker ──
    max_daily_loss_pct: float
    max_drawdown_pct: float
    flatten_on_daily_halt: bool
    flatten_on_drawdown_halt: bool
    drawdown_cooldown_hours: float  # 0 = solo reinicio manual
    max_consecutive_losses: int
    loss_cooldown_minutes: float
    # ── Stops ──
    sl_atr_mult: float
    tp_atr_mult: float             # 0 = sin take-profit
    trail_atr_mult: float          # 0 = sin trailing
    reentry_cooldown_minutes: float
    # ── Calidad de mercado / ejecución ──
    max_spread_bps: float
    max_participation: float       # fracción máxima del volumen de la vela por orden
    min_notional: float            # USDT; Binance exige ~5-10
    # ── Overlay de sentimiento (solo reduce riesgo) ──
    sentiment_min_confidence: float
    sentiment_bearish: float
    sentiment_bullish: float
    sentiment_derisk_factor: float
    state_dir: Path

    @classmethod
    def from_env(cls, market_type: MarketType | None = None, timeframe: str | None = None) -> "RiskSettings":
        base = Settings.from_env() if (market_type is None or timeframe is None) else None
        market = market_type or base.market_type
        tf = timeframe or base.timeframe
        futures = market is MarketType.FUTURE
        state_dir = ROOT_DIR / "data" / "risk"
        state_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            kelly_fraction=_f("RISK_KELLY_FRACTION", "0.25"),
            kelly_min_trades=_i("RISK_KELLY_MIN_TRADES", "30"),
            kelly_window_trades=_i("RISK_KELLY_WINDOW_TRADES", "100"),
            kelly_bootstrap_cap=_f("RISK_KELLY_BOOTSTRAP_CAP", "0.25"),
            kelly_floor=_f("RISK_KELLY_FLOOR", "0.10"),
            max_position_frac=_f("RISK_MAX_POSITION_FRAC", "1.0"),
            max_total_exposure=_f("RISK_MAX_TOTAL_EXPOSURE", "1.0"),
            max_leverage=_f("RISK_MAX_LEVERAGE", "1.0"),
            allow_short=_b("RISK_ALLOW_SHORT", "true" if futures else "false") and futures,
            var_confidence=_f("RISK_VAR_CONFIDENCE", "0.95"),
            var_horizon_bars=_i("RISK_VAR_HORIZON_BARS", str(max(1, round(bars_per_day(tf))))),
            var_limit_pct=_f("RISK_VAR_LIMIT_PCT", "0.03"),
            var_window=_i("RISK_VAR_WINDOW", "720"),
            var_min_obs=_i("RISK_VAR_MIN_OBS", "100"),
            max_daily_loss_pct=_f("RISK_MAX_DAILY_LOSS_PCT", "0.03"),
            max_drawdown_pct=_f("RISK_MAX_DRAWDOWN_PCT", "0.15"),
            flatten_on_daily_halt=_b("RISK_FLATTEN_ON_DAILY_HALT", "false"),
            flatten_on_drawdown_halt=_b("RISK_FLATTEN_ON_DRAWDOWN_HALT", "true"),
            drawdown_cooldown_hours=_f("RISK_DRAWDOWN_COOLDOWN_HOURS", "0"),
            max_consecutive_losses=_i("RISK_MAX_CONSECUTIVE_LOSSES", "5"),
            loss_cooldown_minutes=_f("RISK_LOSS_COOLDOWN_MINUTES", "120"),
            sl_atr_mult=_f("RISK_SL_ATR_MULT", "2.0"),
            tp_atr_mult=_f("RISK_TP_ATR_MULT", "4.0"),
            trail_atr_mult=_f("RISK_TRAIL_ATR_MULT", "2.5"),
            reentry_cooldown_minutes=_f("RISK_REENTRY_COOLDOWN_MINUTES", "60"),
            max_spread_bps=_f("RISK_MAX_SPREAD_BPS", "15"),
            max_participation=_f("RISK_MAX_PARTICIPATION", "0.02"),
            min_notional=_f("RISK_MIN_NOTIONAL", "10"),
            sentiment_min_confidence=_f("RISK_SENTIMENT_MIN_CONFIDENCE", "0.3"),
            sentiment_bearish=_f("RISK_SENTIMENT_BEARISH", "35"),
            sentiment_bullish=_f("RISK_SENTIMENT_BULLISH", "65"),
            sentiment_derisk_factor=_f("RISK_SENTIMENT_DERISK_FACTOR", "0.5"),
            state_dir=state_dir,
        )
