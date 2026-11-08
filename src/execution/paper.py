from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from config.live_settings import ExecutionSettings
from config.rl_settings import CostSettings
from src.rl.costs import CostModel


@dataclass
class PaperPosition:
    qty: float = 0.0
    entry_price: float = 0.0


class PaperGateway:
    """Local spot-like simulator. It never creates an exchange client or sends network orders."""

    def __init__(
        self,
        price_provider: Callable[[str], tuple[float, float]],
        initial_cash: float,
        costs: CostSettings,
        allow_short: bool = False,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("El capital inicial paper debe ser positivo.")
        self.allow_short = allow_short
        self.price_provider = price_provider
        self.cash = float(initial_cash)
        self.cost_model = CostModel(costs)
        self.positions: dict[str, PaperPosition] = {}
        self.closed_fills: list[dict] = []

    def qty(self, symbol: str) -> float:
        return self.positions.get(symbol, PaperPosition()).qty

    def entry_price(self, symbol: str) -> float | None:
        entry = self.positions.get(symbol, PaperPosition()).entry_price
        return entry if entry > 0 else None

    def equity(self, prices: dict[str, float] | None = None) -> float:
        prices = prices or {}
        return self.cash + sum(
            position.qty * prices.get(symbol, self.price_provider(symbol)[0])
            for symbol, position in self.positions.items()
        )

    def rebalance(self, symbol: str, target_weight: float, equity: float, price: float,
                  quote_volume: float) -> dict | None:
        if price <= 0 or equity <= 0:
            raise ValueError("Precio y equity deben ser positivos para simular una orden.")
        position = self.positions.setdefault(symbol, PaperPosition())
        target_qty = target_weight * equity / price
        delta = target_qty - position.qty
        if not self.allow_short and delta < -1e-10 and position.qty + delta < -1e-10:
            raise ValueError("Paper Spot no permite abrir posiciones cortas.")
        if abs(delta * price) < 1e-8:
            return None

        fill = self.cost_model.execute(delta, price, quote_volume)
        if delta > 0:
            affordable = self.cost_model.max_affordable_buy(self.cash, price, quote_volume)
            if delta > affordable:
                fill = self.cost_model.execute(affordable, price, quote_volume)
        old_qty = position.qty
        old_entry = position.entry_price
        realized = 0.0
        if old_qty != 0 and delta != 0 and old_qty * delta < 0:
            closed_qty = min(abs(old_qty), abs(delta))
            direction = 1 if old_qty > 0 else -1
            realized = closed_qty * direction * (fill.price - old_entry)
        self.cash -= fill.qty * fill.price + fill.fee
        new_qty = old_qty + fill.qty
        if abs(new_qty) <= 1e-10:
            new_qty = 0.0
            position.entry_price = 0.0
        elif old_qty == 0 or old_qty * fill.qty > 0:
            position.entry_price = (
                (abs(old_qty) * old_entry + abs(fill.qty) * fill.price) / abs(new_qty)
            )
        elif old_qty * new_qty < 0:
            position.entry_price = fill.price
        position.qty = new_qty
        record = {
            "symbol": symbol,
            "side": "buy" if fill.qty > 0 else "sell",
            "qty": abs(fill.qty),
            "price": fill.price,
            "filled": abs(fill.qty),
            "avg_price": fill.price,
            "fee": fill.fee,
            "realized_pnl": realized - fill.fee,
            "notional": fill.notional,
            "method": "paper_market",
            "orders": 1,
            "status": "filled",
        }
        self.closed_fills.append(record)
        return record


class OrderExecutor:
    """Applies an approved target exposure through the configured gateway."""

    def __init__(self, gateway: PaperGateway, settings: ExecutionSettings) -> None:
        self.gateway = gateway
        self.settings = settings

    async def execute(self, symbol: str, target_weight: float, equity: float,
                      price: float, quote_volume: float) -> dict | None:
        current_weight = self.gateway.qty(symbol) * price / equity if equity > 0 else 0.0
        increases_risk = (
            target_weight * current_weight >= 0
            and abs(target_weight) > abs(current_weight)
        )
        if increases_risk and abs(target_weight - current_weight) < self.settings.min_rebalance:
            return None
        return self.gateway.rebalance(symbol, target_weight, equity, price, quote_volume)
