from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol


class RiskAction(str, Enum):
    ALLOW = "ALLOW"        # se aprueba lo pedido
    REDUCE = "REDUCE"      # se aprueba menos exposición de la pedida
    HOLD = "HOLD"          # no operar (mantener la posición actual)
    BLOCK = "BLOCK"        # se bloquea aumentar riesgo
    FLATTEN = "FLATTEN"    # cerrar toda la posición


class BreakerState(str, Enum):
    OK = "OK"
    DAILY_HALT = "DAILY_HALT"
    DRAWDOWN_HALT = "DRAWDOWN_HALT"
    STREAK_COOLDOWN = "STREAK_COOLDOWN"


class PortfolioSource(str, Enum):
    SIMULATION = "simulation"
    EXCHANGE = "exchange"


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Foto de la cartera en el momento de decidir. Pesos = fracción del equity (con signo)."""
    symbol: str
    equity: float
    weight: float                 # exposición actual del símbolo
    price: float                  # último precio
    timestamp: datetime           # UTC, timezone-aware
    entry_price: float | None = None   # precio medio de entrada de la posición abierta
    other_exposure: float = 0.0   # suma de |exposición| de los demás símbolos
    source: PortfolioSource = PortfolioSource.SIMULATION


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    price: float
    atr_pct: float                # ATR / precio (de FEATURE_COLUMNS)
    high: float | None = None     # extremos de la última vela (para stops intravela)
    low: float | None = None
    spread_bps: float | None = None
    bar_quote_volume: float | None = None   # volumen de la vela en USDT


class SentimentLike(Protocol):
    """Compatible con SentimentResult y SentimentEvent de la Fase 2."""
    score: float
    confidence: float
    degraded: bool


@dataclass(frozen=True, slots=True)
class RiskReason:
    code: str
    message: str
    before: float
    after: float


@dataclass(frozen=True, slots=True)
class RiskDecision:
    symbol: str
    timestamp: datetime
    requested_weight: float
    approved_weight: float
    current_weight: float
    action: RiskAction
    reasons: tuple[RiskReason, ...] = ()
    stop_price: float | None = None
    take_profit_price: float | None = None
    breaker: BreakerState = BreakerState.OK
    kelly_cap: float | None = None
    var_cap: float | None = None
    var_unit: float | None = None      # VaR por unidad de exposición (fracción)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(r.code for r in self.reasons)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(), "symbol": self.symbol,
            "action": self.action.value, "requested": round(self.requested_weight, 4),
            "approved": round(self.approved_weight, 4), "current": round(self.current_weight, 4),
            "reasons": [{"code": r.code, "msg": r.message, "before": round(r.before, 4),
                         "after": round(r.after, 4)} for r in self.reasons],
            "stop": self.stop_price, "take_profit": self.take_profit_price,
            "breaker": self.breaker.value, "kelly_cap": self.kelly_cap,
            "var_cap": self.var_cap, "var_unit": self.var_unit,
        }
