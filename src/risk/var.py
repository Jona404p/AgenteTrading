from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True, slots=True)
class VaRResult:
    var: float          # pérdida VaR por unidad de exposición, al horizonte (fracción ≥ 0)
    cvar: float         # expected shortfall al horizonte
    n_obs: int
    method: str         # "empirical" | "atr_fallback"


def historical_var(losses: np.ndarray, confidence: float) -> float:
    return max(0.0, float(np.quantile(losses, confidence)))


def expected_shortfall(losses: np.ndarray, confidence: float) -> float:
    v = np.quantile(losses, confidence)
    tail = losses[losses >= v]
    return max(0.0, float(tail.mean())) if len(tail) else max(0.0, float(v))


def cornish_fisher_var(losses: np.ndarray, confidence: float) -> float:
    """VaR paramétrico con corrección de asimetría y curtosis (colas gruesas). Ignora la media (conservador)."""
    s = losses.std(ddof=1)
    if s <= 0:
        return 0.0
    x = (losses - losses.mean()) / s
    skew, kurt = float((x ** 3).mean()), float((x ** 4).mean() - 3.0)
    z = NormalDist().inv_cdf(confidence)
    z_cf = (z + (z ** 2 - 1) * skew / 6 + (z ** 3 - 3 * z) * kurt / 24
            - (2 * z ** 3 - 5 * z) * skew ** 2 / 36)
    return max(0.0, float(s * z_cf))


class VaRModel:
    """VaR/CVaR por unidad de exposición. Usa el mayor de histórico y Cornish-Fisher, escalado
    por √horizonte. Si hay pocos datos cae a una aproximación con ATR (conservadora)."""

    def __init__(self, confidence: float, horizon_bars: int, window: int, min_obs: int) -> None:
        self.confidence, self.horizon = confidence, max(1, horizon_bars)
        self.window, self.min_obs = window, min_obs

    def compute(self, returns, side: int, fallback_atr_pct: float) -> VaRResult:
        """`side` = +1 para largos (pierden si baja) y −1 para cortos (pierden si sube)."""
        scale = math.sqrt(self.horizon)
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)][-self.window:]
        if len(r) < self.min_obs:
            z = NormalDist().inv_cdf(self.confidence)
            v = z * fallback_atr_pct * scale
            return VaRResult(v, v * 1.25, len(r), "atr_fallback")
        losses = -side * r
        var1 = max(historical_var(losses, self.confidence), cornish_fisher_var(losses, self.confidence))
        es1 = max(expected_shortfall(losses, self.confidence), var1)
        return VaRResult(var1 * scale, es1 * scale, len(r), "empirical")


def max_weight_for_var(limit_pct: float, var_unit: float) -> float:
    """Mayor |exposición| tal que exposición · VaR_unidad ≤ límite."""
    return float("inf") if var_unit <= 1e-12 else limit_pct / var_unit
