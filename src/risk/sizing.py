from __future__ import annotations

from collections import deque

import numpy as np


def kelly_binary(win_rate: float, payoff: float) -> float:
    """Kelly para apuestas binarias: f* = p − (1−p)/b, con b = ganancia media / pérdida media."""
    if payoff <= 0:
        return 0.0
    return win_rate - (1.0 - win_rate) / payoff


def kelly_continuous(returns: np.ndarray) -> float:
    """Kelly continuo f* = μ/σ² a partir de retornos por periodo."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    var = r.var(ddof=1)
    return float(r.mean() / var) if var > 1e-18 else 0.0


class TradeStats:
    """Ventana móvil de operaciones cerradas (retorno neto de cada una, fracción de la entrada)."""

    def __init__(self, window: int = 100) -> None:
        self._pnl: deque[float] = deque(maxlen=window)

    def add(self, pnl_pct: float) -> None:
        self._pnl.append(float(pnl_pct))

    def __len__(self) -> int:
        return len(self._pnl)

    @property
    def consecutive_losses(self) -> int:
        n = 0
        for p in reversed(self._pnl):
            if p < 0:
                n += 1
            else:
                break
        return n

    def kelly(self) -> float | None:
        wins = [p for p in self._pnl if p > 0]
        losses = [-p for p in self._pnl if p < 0]
        if not self._pnl or not losses:
            # sin pérdidas observadas el Kelly no está acotado: no es una estimación fiable
            return None
        if not wins:
            return -1.0
        payoff = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
        return kelly_binary(len(wins) / len(self._pnl), payoff)

    def to_list(self) -> list[float]:
        return list(self._pnl)


class KellySizer:
    """Tope de exposición por Kelly fraccional.

    · Con ≥ `min_trades` operaciones: cap = clip(fracción · f*, floor, max_cap).
    · Con menos: cap = bootstrap_cap (conservador). Sin ventaja (f* ≤ 0) se reduce a `floor`
      —no a cero— para poder seguir generando estadística y recuperarse.
    """

    def __init__(self, fraction: float, min_trades: int, bootstrap_cap: float,
                 floor: float, max_cap: float) -> None:
        self.fraction, self.min_trades = fraction, min_trades
        self.bootstrap_cap, self.floor, self.max_cap = bootstrap_cap, floor, max_cap

    def cap(self, stats: TradeStats) -> tuple[float, str]:
        if len(stats) < self.min_trades:
            return min(self.bootstrap_cap, self.max_cap), "bootstrap"
        f = stats.kelly()
        if f is None:
            return min(self.bootstrap_cap, self.max_cap), "bootstrap"
        if f <= 0:
            return min(self.floor, self.max_cap), "no_edge"
        return float(np.clip(self.fraction * f, self.floor, self.max_cap)), "kelly"
