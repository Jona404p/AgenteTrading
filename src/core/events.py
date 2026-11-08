from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KlineEvent:
    symbol: str          # formato ccxt: "BTC/USDT"
    timeframe: str
    open_time: int       # ms epoch UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool      # True solo cuando la vela cerró


@dataclass(frozen=True, slots=True)
class OrderBookEvent:
    symbol: str
    timestamp: int                          # ms locales de recepción
    bids: tuple[tuple[float, float], ...]   # (precio, cantidad), mejor primero
    asks: tuple[tuple[float, float], ...]

    @property
    def best_bid(self) -> float:
        return self.bids[0][0]

    @property
    def best_ask(self) -> float:
        return self.asks[0][0]

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread_bps(self) -> float:
        return (self.best_ask - self.best_bid) / self.mid_price * 1e4

    def imbalance(self, depth: int = 10) -> float:
        """(bid_vol - ask_vol) / (bid_vol + ask_vol) en los primeros `depth` niveles. Rango [-1, 1]."""
        b = sum(q for _, q in self.bids[:depth])
        a = sum(q for _, q in self.asks[:depth])
        total = b + a
        return (b - a) / total if total else 0.0


@dataclass(frozen=True, slots=True)
class SentimentEvent:
    asset: str                 # activo base: "BTC", "ETH"...
    score: float               # 0 = extremadamente bajista, 50 = neutral, 100 = extremadamente alcista
    confidence: float          # 0..1
    n_scored: int              # titulares efectivamente puntuados
    fear_greed: int | None     # Fear & Greed Index (0-100) o None si no disponible
    degraded: bool             # True si el LLM/fuentes fallaron (score neutral por defecto)
    timestamp: int             # ms epoch UTC
    drivers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RiskEvent:
    symbol: str                # formato ccxt: "BTC/USDT"
    action: str                # ALLOW | REDUCE | BLOCK | FLATTEN | HOLD
    requested_weight: float    # exposición pedida por el agente (fracción del equity, con signo)
    approved_weight: float     # exposición aprobada tras los filtros de riesgo
    reasons: tuple[str, ...]   # códigos de las reglas que actuaron
    breaker: str               # OK | DAILY_HALT | DRAWDOWN_HALT | STREAK_COOLDOWN
    timestamp: int             # ms epoch UTC


@dataclass(frozen=True, slots=True)
class DriftEvent:
    bar: str
    symbol: str
    level: str
    n_drifted: int
    n_features: int
    drifted: tuple[str, ...]
    reasons: tuple[str, ...]
    tstat: float | None
    drawdown: float


@dataclass(frozen=True, slots=True)
class ExecutionReportEvent:
    symbol: str
    market_id: str
    order_id: int
    client_order_id: str
    side: str
    execution_type: str
    order_status: str
    last_filled_quantity: float
    last_filled_price: float
    cumulative_filled_quantity: float
    cumulative_quote_quantity: float
    commission: float
    commission_asset: str | None
    trade_id: int
    transaction_time: int
    realized_pnl: float

    @property
    def is_fill(self) -> bool:
        return self.execution_type == "TRADE" and self.last_filled_quantity > 0


Handler = Callable[[Any], Awaitable[None] | None]


class EventBus:
    """Bus de eventos mínimo. Acepta handlers síncronos y asíncronos."""

    def __init__(self) -> None:
        self._subs: dict[type, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Handler) -> None:
        self._subs[event_type].append(handler)

    def unsubscribe(self, event_type: type, handler: Handler) -> None:
        if handler in self._subs[event_type]:
            self._subs[event_type].remove(handler)

    async def publish(self, event: Any) -> None:
        for handler in list(self._subs.get(type(event), [])):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # un observer roto no debe tumbar el stream
                log.exception("Error en handler %s", getattr(handler, "__name__", handler))
