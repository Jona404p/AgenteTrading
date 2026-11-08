"""Replay del TradingEnv (F3) con el RiskManager (F4) en medio: comprueba que los límites se respetan de verdad."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
pytest.importorskip("gymnasium")

from config.risk_settings import RiskSettings
from config.rl_settings import RLSettings
from config.settings import MarketType
from src.data.features import FeatureEngineer
from src.risk import BreakerState, MarketSnapshot, PortfolioState, RiskManager
from src.rl.env import TradingEnv
from src.rl.metrics import compute_metrics
from src.rl.splits import FeatureScaler


def _crashy_df(n=2500, seed=3):
    rng = np.random.default_rng(seed)
    drift = np.where((np.arange(n) // 400) % 2 == 0, 0.0004, -0.0012)     # alternan subidas y caídas fuertes
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.012, n)))
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.004,
                         "low": np.minimum(open_, close) * 0.996, "close": close,
                         "volume": rng.uniform(5000, 9000, n)}, index=idx)


def _run(use_risk: bool, tmp_path, seed=11):
    raw = _crashy_df()
    feats = FeatureEngineer().transform(raw)
    rl = replace(RLSettings.from_env(MarketType.SPOT), episode_len=0, min_rebalance=0.02)
    scaler = FeatureScaler(FeatureEngineer.FEATURE_COLUMNS).fit(feats.iloc[:800])
    env = TradingEnv(feats, scaler, rl, random_start=False)
    env.reset()
    rcfg = replace(RiskSettings.from_env(MarketType.SPOT, "1h"), state_dir=tmp_path,
                   kelly_bootstrap_cap=0.6, max_drawdown_pct=0.20, flatten_on_daily_halt=False)
    rm = RiskManager(rcfg, persist=False, require_exchange_state=False)
    rng = np.random.default_rng(seed)
    log_ret = np.log(feats["close"]).diff().to_numpy()
    breaker_states, max_w, done, prev_qty = set(), 0.0, False, 0.0
    entry_px = 0.0
    while not done:
        t = env.t
        action = rng.uniform(-0.3, 1.0)                  # agente "ruidoso" y sesgado a largo
        if use_risk:
            row = feats.iloc[t]
            d = rm.evaluate(
                PortfolioState("BTC/USDT", env.equity, env.weight, row["close"], feats.index[t].to_pydatetime(),
                               env.avg_entry or None),
                MarketSnapshot(row["close"], row["atr_pct"], row["high"], row["low"],
                               bar_quote_volume=row["volume"] * row["close"]),
                action, log_ret[max(0, t - 720):t + 1], None)
            action = d.approved_weight
            breaker_states.add(d.breaker)
        _, _, term, trunc, _ = env.step(np.array([action]))
        done = term or trunc
        max_w = max(max_w, abs(env.weight))
        # notificar aperturas/cierres al gestor de riesgo
        price = feats["close"].iloc[env.t]
        if use_risk:
            if prev_qty == 0 and env.qty != 0:
                rm.on_position_opened("BTC/USDT", 1, env.avg_entry, feats["atr_pct"].iloc[env.t]); entry_px = env.avg_entry
            elif prev_qty != 0 and env.qty == 0:
                rm.on_position_closed("BTC/USDT", 1, entry_px, price, feats.index[env.t].to_pydatetime())
        prev_qty = env.qty
    s = env.episode_stats()
    return compute_metrics(s["equity"], s["bench"], "1h", s["fees_paid"], s["n_trades"], s["turnover"]), max_w, breaker_states


def test_risk_layer_limits_drawdown_and_exposure(tmp_path):
    free, w_free, _ = _run(False, tmp_path)
    guarded, w_guard, states = _run(True, tmp_path)
    assert w_guard <= 1.0 + 1e-6                                   # nunca apalancado en spot
    assert guarded["max_drawdown"] > free["max_drawdown"]          # (negativo) → caída máxima menor con riesgo
    assert guarded["total_return"] > free["total_return"]
