from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        parts = []
        for name in ("train", "val", "test"):
            d = getattr(self, name)
            parts.append(f"{name}: {len(d)} velas [{d.index[0]:%Y-%m-%d} → {d.index[-1]:%Y-%m-%d}]")
        return " | ".join(parts)


def temporal_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15,
                   embargo: int = 0) -> DataSplit:
    """Split cronológico (sin barajar). `embargo` descarta velas entre tramos para que las
    ventanas de observación y los indicadores de cola no se solapen entre train/val/test."""
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac y val_frac deben sumar menos de 1")
    if not df.index.is_monotonic_increasing:
        raise ValueError("El índice debe estar ordenado cronológicamente")
    n = len(df)
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))
    train = df.iloc[:i_tr]
    val = df.iloc[i_tr + embargo:i_va]
    test = df.iloc[i_va + embargo:]
    if min(len(train), len(val), len(test)) < 50:
        raise ValueError(f"Datos insuficientes para el split ({len(train)}/{len(val)}/{len(test)} velas)")
    return DataSplit(train, val, test)


class FeatureScaler:
    """Estandarización robusta (mediana / IQR·0.7413) ajustada SOLO con train y recortada a ±clip.

    Más resistente a colas gruesas que media/desviación (cripto). Se guarda en JSON junto
    al modelo: en producción (F5) debe aplicarse exactamente el mismo scaler.
    """

    def __init__(self, columns: list[str], clip: float = 5.0) -> None:
        self.columns = list(columns)
        self.clip = clip
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.center is not None

    def fit(self, train: pd.DataFrame) -> "FeatureScaler":
        x = train[self.columns].to_numpy(dtype=np.float64)
        self.center = np.median(x, axis=0)
        q75, q25 = np.percentile(x, [75, 25], axis=0)
        scale = (q75 - q25) * 0.7413          # ≈ desviación típica si fuera normal
        fallback = np.std(x, axis=0)
        scale = np.where(scale > 1e-12, scale, fallback)
        self.scale = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform_array(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler no ajustado: llama a fit(train) primero")
        z = (x - self.center) / self.scale
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.transform_array(df[self.columns].to_numpy(dtype=np.float64))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "columns": self.columns, "clip": self.clip,
            "center": self.center.tolist(), "scale": self.scale.tolist(),
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        d = json.loads(path.read_text())
        s = cls(d["columns"], d["clip"])
        s.center, s.scale = np.array(d["center"]), np.array(d["scale"])
        return s
