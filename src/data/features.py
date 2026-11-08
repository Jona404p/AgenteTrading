from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import pandas_ta_classic as ta   # fork mantenido (recomendado)
except ImportError:
    import pandas_ta as ta           # librería original, si la tienes instalada


@dataclass(frozen=True)
class FeatureConfig:
    rsi_length: int = 14
    ema_fast: int = 20
    ema_slow: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_length: int = 20
    bb_std: float = 2.0
    atr_length: int = 14
    adx_length: int = 14
    vol_window: int = 24
    zscore_window: int = 48


def _pick(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """Selecciona la primera columna por prefijo (robusto a cambios de nombre entre versiones)."""
    col = next(c for c in frame.columns if str(c).startswith(prefix))
    return frame[col]


class FeatureEngineer:
    FEATURE_COLUMNS = [
        "log_return", "hl_range_pct", "rsi", "macd", "macd_signal", "macd_hist",
        "ema_ratio", "dist_ema_slow", "bb_bandwidth", "bb_percent",
        "atr_pct", "adx", "volatility", "volume_zscore",
    ]

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.cfg = config or FeatureConfig()

    @property
    def warmup_periods(self) -> int:
        c = self.cfg
        return max(
            c.ema_slow, c.macd_slow + c.macd_signal, c.bb_length,
            c.atr_length, c.adx_length * 2, c.vol_window + 1, c.zscore_window,
        )

    def transform(self, ohlcv: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """Recibe OHLCV (índice UTC) y devuelve OHLCV + features. Solo usa información pasada."""
        c = self.cfg
        df = ohlcv.copy()
        close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

        df["log_return"] = np.log(close / close.shift(1))
        df["hl_range_pct"] = (high - low) / close
        df["rsi"] = ta.rsi(close, length=c.rsi_length)

        macd = ta.macd(close, fast=c.macd_fast, slow=c.macd_slow, signal=c.macd_signal)
        # MACD viene en unidades de precio: se divide por el cierre para que sea comparable
        # entre épocas y activos (estacionariedad).
        df["macd"] = _pick(macd, "MACD_") / close
        df["macd_hist"] = _pick(macd, "MACDh_") / close
        df["macd_signal"] = _pick(macd, "MACDs_") / close

        ema_f = ta.ema(close, length=c.ema_fast)
        ema_s = ta.ema(close, length=c.ema_slow)
        df["ema_ratio"] = ema_f / ema_s - 1
        df["dist_ema_slow"] = close / ema_s - 1

        bb = ta.bbands(close, length=c.bb_length, std=c.bb_std)
        df["bb_bandwidth"] = _pick(bb, "BBB_")
        df["bb_percent"] = _pick(bb, "BBP_")

        df["atr_pct"] = ta.atr(high, low, close, length=c.atr_length) / close
        df["adx"] = _pick(ta.adx(high, low, close, length=c.adx_length), "ADX_")

        df["volatility"] = df["log_return"].rolling(c.vol_window).std()
        roll = vol.rolling(c.zscore_window)
        df["volume_zscore"] = (vol - roll.mean()) / roll.std()

        df = df.replace([np.inf, -np.inf], np.nan)
        return df.dropna() if dropna else df
