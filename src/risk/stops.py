from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StopTrigger:
    kind: str            # "stop_loss" | "trailing_stop" | "take_profit"
    price: float         # precio de disparo (nivel), no el de ejecución real


class PositionStops:
    """SL / TP / trailing basados en ATR para una posición abierta.

    Las distancias se fijan con el ATR de la ENTRADA. El trailing sigue al mejor precio alcanzado
    y nunca retrocede. Si en una misma vela se tocan SL y TP se asume el SL (caso conservador).
    """

    def __init__(self, side: int, entry: float, atr_pct: float,
                 sl_mult: float, tp_mult: float, trail_mult: float) -> None:
        if side not in (1, -1):
            raise ValueError("side debe ser +1 (largo) o −1 (corto)")
        self.side, self.entry = side, entry
        dist = max(atr_pct, 1e-6) * entry
        self.stop = self.initial_stop = entry - side * sl_mult * dist
        self.take_profit = entry + side * tp_mult * dist if tp_mult > 0 else None
        self.trail_dist = trail_mult * dist if trail_mult > 0 else None
        self.extreme = entry       # máximo (largo) / mínimo (corto) desde la entrada

    def _breached(self, level: float, high: float, low: float) -> bool:
        return low <= level if self.side == 1 else high >= level

    def update(self, high: float, low: float) -> StopTrigger | None:
        # 1) ¿se tocó un nivel vigente en esta vela? (se evalúa ANTES de mover el trailing)
        if self._breached(self.stop, high, low):
            return StopTrigger("stop_loss" if self.stop == self.initial_stop else "trailing_stop", self.stop)
        if self.take_profit is not None and (
                high >= self.take_profit if self.side == 1 else low <= self.take_profit):
            return StopTrigger("take_profit", self.take_profit)
        # 2) subir/bajar el trailing con el nuevo extremo
        self.extreme = max(self.extreme, high) if self.side == 1 else min(self.extreme, low)
        if self.trail_dist is not None:
            new = self.extreme - self.side * self.trail_dist
            if (new > self.stop) if self.side == 1 else (new < self.stop):
                self.stop = new
        return None
