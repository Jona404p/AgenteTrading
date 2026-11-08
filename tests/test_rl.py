import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
pytest.importorskip("gymnasium")

from config.rl_settings import CostSettings, RLSettings
from config.settings import MarketType
from src.data.features import FeatureEngineer
from src.rl.costs import CostModel
from src.rl.env import TradingEnv
from src.rl.metrics import bars_per_year, compute_metrics
from src.rl.splits import FeatureScaler, temporal_split
from src.rl.training import _validate_training_data


def _df(n=1500, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.003,
                         "low": np.minimum(open_, close) * 0.997, "close": close,
                         "volume": rng.uniform(1000, 2000, n)}, index=idx)


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("BINANCE_ENV", "testnet")
    cfg = RLSettings.from_env(MarketType.SPOT)
    cfg = replace(cfg, episode_len=0)
    feats = FeatureEngineer().transform(_df())
    scaler = FeatureScaler(FeatureEngineer.FEATURE_COLUMNS).fit(feats.iloc[:800])
    return cfg, feats, scaler


def test_cost_model_buy_sell_adverse():
    cm = CostModel(CostSettings(taker_fee_bps=10, base_slippage_bps=2, impact_coef_bps=0))
    b, s = cm.execute(1, 100, 1e9), cm.execute(-1, 100, 1e9)
    assert b.price == pytest.approx(100.02) and s.price == pytest.approx(99.98)
    assert b.fee == pytest.approx(100.02 * 0.001)
    big = cm.execute(1, 100, 1e9)
    assert CostModel(CostSettings(impact_coef_bps=50)).execute(1000, 100, 1e5).price > big.price


def test_affordable_buy_never_overspends():
    cm = CostModel(CostSettings())
    q = cm.max_affordable_buy(1000, 50, 1e6)
    f = cm.execute(q, 50, 1e6)
    assert f.notional + f.fee <= 1000 + 1e-9 and f.notional + f.fee > 990


def test_split_is_chronological_with_embargo():
    df = _df()
    s = temporal_split(df, 0.7, 0.15, embargo=10)
    assert s.train.index[-1] < s.val.index[0] < s.val.index[-1] < s.test.index[0]
    assert len(df) - (len(s.train) + len(s.val) + len(s.test)) == 20


def test_training_data_validation_rejects_duplicate_timestamps(setup):
    _, feats, _ = setup
    invalid = pd.concat([feats, feats.iloc[[-1]]]).sort_index()

    with pytest.raises(ValueError, match="ordenados y sin timestamps duplicados"):
        _validate_training_data(invalid, "BTC/USDT")


def test_training_data_validation_rejects_non_finite_features(setup):
    _, feats, _ = setup
    invalid = feats.iloc[:1000].copy()
    invalid.iloc[200, invalid.columns.get_loc("rsi")] = np.nan

    with pytest.raises(ValueError, match="NaN o valores no finitos"):
        _validate_training_data(invalid, "BTC/USDT")


def test_scaler_uses_train_only_and_roundtrips(tmp_path):
    feats = FeatureEngineer().transform(_df())
    train = feats.iloc[:500]
    sc = FeatureScaler(FeatureEngineer.FEATURE_COLUMNS).fit(train)
    center = sc.center.copy()
    sc.transform(feats.iloc[500:])           # transformar otros datos no cambia el ajuste
    assert np.allclose(sc.center, center)
    p = tmp_path / "s.json"; sc.save(p)
    sc2 = FeatureScaler.load(p)
    assert np.allclose(sc.transform(feats.iloc[:10]), sc2.transform(feats.iloc[:10]))
    assert np.abs(sc.transform(feats)).max() <= 5.0


def test_env_gym_api_and_obs_shape(setup):
    cfg, feats, sc = setup
    env = TradingEnv(feats, sc, cfg, random_start=False)
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()
    obs, r, term, trunc, info = env.step(np.array([0.5], dtype=np.float32))
    assert obs.shape == env.observation_space.shape and isinstance(r, float)


def test_flat_agent_keeps_cash(setup):
    cfg, feats, sc = setup
    env = TradingEnv(feats, sc, cfg, random_start=False)
    env.reset()
    done = False
    while not done:
        _, r, term, trunc, _ = env.step(np.array([0.0]))
        done = term or trunc
    assert env.equity == pytest.approx(cfg.initial_cash) and env.n_trades == 0


def test_long_only_ignores_negative_actions(setup):
    cfg, feats, sc = setup
    env = TradingEnv(feats, sc, cfg, random_start=False)
    env.reset()
    for _ in range(20):
        env.step(np.array([-1.0]))
    assert env.qty == 0.0


def test_costs_reduce_equity_vs_frictionless(setup):
    """Mismo comportamiento con y sin costes: con costes debe terminar con menos equity."""
    cfg, feats, sc = setup
    free = replace(cfg, costs=CostSettings(0, 0, 0, 0, 0))
    def run(c):
        env = TradingEnv(feats, sc, c, random_start=False); env.reset()
        rng = np.random.default_rng(3); done = False
        while not done:
            _, _, te, tr, _ = env.step(rng.uniform(-1, 1, 1)); done = te or tr
        return env
    e_cost, e_free = run(cfg), run(free)
    assert e_cost.fees_paid > 0 and e_cost.equity < e_free.equity


def test_no_lookahead_fill_uses_next_open(setup):
    cfg, feats, sc = setup
    cfg = replace(cfg, costs=CostSettings(0, 0, 0, 0, 0), min_rebalance=0.0)
    env = TradingEnv(feats, sc, cfg, random_start=False)
    env.reset(); t0 = env.t
    env.step(np.array([1.0]))
    # compra completa a open[t0+1] sin costes → qty = cash / open[t0+1]
    assert env.qty == pytest.approx(cfg.initial_cash / env.open[t0 + 1], rel=1e-6)


def test_short_only_when_allowed(setup):
    cfg, feats, sc = setup
    cfg = replace(cfg, allow_short=True, min_rebalance=0.0)
    env = TradingEnv(feats, sc, cfg, random_start=False)
    env.reset(); env.step(np.array([-1.0]))
    assert env.qty < 0 and env.weight == pytest.approx(-1.0, abs=0.05)


def test_ruin_terminates(setup):
    cfg, feats, sc = setup
    cfg = replace(cfg, ruin_equity_frac=0.9999, min_rebalance=0.0)
    env = TradingEnv(feats, sc, cfg, random_start=False); env.reset()
    term = False
    for _ in range(200):
        _, _, term, trunc, _ = env.step(np.array([1.0]))   # solo comisiones ya lo hunden
        if term or trunc: break
    assert term


def test_random_start_episode_length(setup):
    cfg, feats, sc = setup
    cfg = replace(cfg, episode_len=100)
    env = TradingEnv(feats, sc, cfg, random_start=True)
    env.reset(seed=5); n = 0; done = False
    while not done:
        _, _, te, tr, _ = env.step(np.array([0.0])); n += 1; done = te or tr
    assert n == 100


def test_metrics_basic():
    eq = np.array([100, 101, 99, 103.0])
    m = compute_metrics(eq, eq, "1h")
    assert m["total_return"] == pytest.approx(0.03) and m["max_drawdown"] < 0
    assert bars_per_year("1h") == pytest.approx(8760)
