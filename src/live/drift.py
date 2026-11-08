from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.live_settings import DriftSettings


def gate_exposure_on_critical(current: float, requested: float) -> float:
    """Under CRITICAL drift, permit only de-risking to flat or a smaller same-side position."""
    if not np.isfinite(current) or not np.isfinite(requested):
        raise ValueError("La exposición de riesgo debe ser finita.")
    if abs(current) <= 1e-12 or current * requested < 0:
        return 0.0
    return float(np.copysign(min(abs(current), abs(requested)), current))


@dataclass(frozen=True)
class DriftStatus:
    level: str
    enabled: bool
    ready: bool
    n_drifted: int
    n_features: int
    drifted: tuple[str, ...]
    reasons: tuple[str, ...]
    drawdown: float
    psi_max: float | None

    @property
    def entries_paused(self) -> bool:
        return self.level == "CRITICAL"

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "enabled": self.enabled,
            "ready": self.ready,
            "n_drifted": self.n_drifted,
            "n_features": self.n_features,
            "drifted": list(self.drifted),
            "reasons": list(self.reasons),
            "drawdown": self.drawdown,
            "psi_max": self.psi_max,
            "entries_paused": self.entries_paused,
        }


class ModelDriftMonitor:
    def __init__(
        self,
        settings: DriftSettings,
        columns: list[str],
        reference: np.ndarray | None,
        validation_drawdown: float | None = None,
        window: int = 50,
    ) -> None:
        if window < 20:
            raise ValueError("La ventana de drift debe tener al menos 20 observaciones.")
        self.settings = settings
        self.columns = list(columns)
        self.reference = np.asarray(reference, dtype=float) if reference is not None else None
        self.validation_drawdown = validation_drawdown
        self.window = window
        self._recent: list[np.ndarray] = []

    def evaluate(self, row: np.ndarray, drawdown: float) -> DriftStatus:
        values = np.asarray(row, dtype=float).reshape(-1)
        if values.size != len(self.columns) or not np.isfinite(values).all():
            raise ValueError("La observación de drift no coincide con las features ni es finita.")
        self._recent.append(values)
        self._recent = self._recent[-self.window:]
        if not self.settings.enabled:
            return DriftStatus("DISABLED", False, False, 0, len(self.columns), (), ("monitor desactivado",), drawdown, None)
        if self.reference is None or self.reference.ndim != 2 or self.reference.shape[1] != len(self.columns):
            return DriftStatus("UNAVAILABLE", True, False, 0, len(self.columns), (), ("referencia de entrenamiento no disponible",), drawdown, None)
        if len(self.reference) < 20:
            return DriftStatus("UNAVAILABLE", True, False, 0, len(self.columns), (), ("referencia de entrenamiento insuficiente",), drawdown, None)
        if len(self._recent) < self.window:
            return DriftStatus(
                "WARMUP", True, False, 0, len(self.columns), (),
                (f"esperando {self.window - len(self._recent)} observaciones para PSI",), drawdown, None,
            )

        baseline = self.reference[-min(len(self.reference), 5000):]
        current = np.asarray(self._recent)
        psi_by_feature: dict[str, float] = {}
        for index, name in enumerate(self.columns):
            edges = np.unique(np.quantile(baseline[:, index], np.linspace(0, 1, 11)))
            if len(edges) < 3:
                continue
            edges[0], edges[-1] = -np.inf, np.inf
            expected = np.histogram(baseline[:, index], bins=edges)[0].astype(float)
            observed = np.histogram(current[:, index], bins=edges)[0].astype(float)
            expected = np.maximum(expected / max(expected.sum(), 1), 1e-6)
            observed = np.maximum(observed / max(observed.sum(), 1), 1e-6)
            psi_by_feature[name] = float(np.sum((observed - expected) * np.log(observed / expected)))

        drifted = tuple(name for name, value in psi_by_feature.items() if value >= self.settings.psi_threshold)
        psi_max = max(psi_by_feature.values(), default=0.0)
        critical_by_psi = (
            psi_max >= self.settings.critical_psi_threshold
            or (len(drifted) / max(len(self.columns), 1) >= self.settings.critical_feature_fraction)
        )
        critical_by_drawdown = drawdown >= self.settings.max_drawdown
        if critical_by_psi or critical_by_drawdown:
            level = "CRITICAL"
        elif drifted:
            level = "WARN"
        else:
            level = "OK"
        reasons: list[str] = []
        if critical_by_psi:
            reasons.append(
                f"drift crítico: PSI máximo {psi_max:.3f}, {len(drifted)}/{len(self.columns)} features"
            )
        elif drifted:
            reasons.append(f"drift de features: {len(drifted)}/{len(self.columns)} sobre PSI {self.settings.psi_threshold:.3f}")
        if critical_by_drawdown:
            reasons.append(f"drawdown {drawdown:.1%} >= límite {self.settings.max_drawdown:.1%}")
        if not reasons:
            reasons.append("features dentro de la distribución de entrenamiento")
        return DriftStatus(
            level, True, True, len(drifted), len(self.columns), drifted,
            tuple(reasons), drawdown, psi_max,
        )


__all__ = ["DriftStatus", "ModelDriftMonitor", "gate_exposure_on_critical"]
