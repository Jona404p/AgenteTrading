"""Entrena el agente RL (Fase 3).

Uso:
    python scripts/train_rl.py --symbol BTC/USDT [--timesteps 300000] [--algo ppo]
    python scripts/train_rl.py --synthetic --timesteps 20000     # prueba de humo sin datos reales
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.rl_settings import RLSettings
from config.settings import Settings
from src.core.logger import setup_logging
from src.data.features import FeatureEngineer
from src.rl.training import load_features, train


def synthetic_ohlcv(n: int = 12000, seed: int = 0) -> pd.DataFrame:
    """Serie sintética con regímenes de tendencia/rango (solo para validar el pipeline)."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([-0.0004, 0.0, 0.0006], size=n // 300 + 1), 300)[:n]
    close = 30000 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0005, n))
    spread = np.abs(rng.normal(0.003, 0.001, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + spread),
                         "low": np.minimum(open_, close) * (1 - spread), "close": close,
                         "volume": rng.uniform(200, 800, n)}, index=idx)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default=None)
    p.add_argument("--algo", choices=["ppo", "a2c", "sac"], default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()

    st = Settings.from_env()
    setup_logging(st.log_level)
    cfg = RLSettings.from_env(st.market_type)
    if a.algo:
        cfg = replace(cfg, algo=a.algo)
    symbol = a.symbol or st.symbols[0]

    if a.synthetic:
        df = FeatureEngineer().transform(synthetic_ohlcv())
        symbol = "SYNTH/USDT"
    else:
        df = load_features(symbol, st.timeframe, st.processed_dir, st.raw_dir)

    res = train(df, cfg, st.timeframe, symbol, total_timesteps=a.timesteps)
    print(f"\nModelo guardado en: {res.run_dir}\n{res.split_summary}")
    for name, m in (("VALIDACIÓN", res.val_metrics), ("TEST (fuera de muestra)", res.test_metrics)):
        print(f"\n== {name} ==")
        for k, v in m.items():
            print(f"  {k:>16}: {v:.4f}" if isinstance(v, float) else f"  {k:>16}: {v}")


if __name__ == "__main__":
    main()
