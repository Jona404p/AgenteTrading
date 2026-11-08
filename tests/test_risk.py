import asyncio
import json
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.risk_settings import RiskSettings, bars_per_day
from config.settings import MarketType
from src.core.events import EventBus, RiskEvent
from src.risk import (BreakerState, CircuitBreaker, KellySizer, MarketSnapshot, PortfolioState,
                      PortfolioSource, PositionStops, RiskAction, RiskManager, TradeStats,
                      VaRModel, kelly_binary, kelly_continuous)
from src.risk.var import cornish_fisher_var, historical_var, max_weight_for_var

T0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path):
    c = RiskSettings.from_env(MarketType.SPOT, "1h")
    return replace(c, state_dir=tmp_path)


def rets(n=500, sigma=0.004, seed=0):
    return np.random.default_rng(seed).normal(0, sigma, n)


def mk(cfg, **kw):
    return RiskManager(
        cfg, persist=kw.pop("persist", False), require_exchange_state=False, **kw
    )


def state(w=0.0, equity=10_000.0, price=100.0, t=T0, entry=None, others=0.0):
    return PortfolioState("BTC/USDT", equity, w, price, t, entry, others)


def mkt(**kw):
    d = dict(price=100.0, atr_pct=0.01, high=100.5, low=99.5)
    d.update(kw)
    return MarketSnapshot(**d)


@dataclass
class Sent:
    score: float
    confidence: float = 0.8
    degraded: bool = False


# ───────────────────────────── sizing
def test_kelly_formulas():
    assert kelly_binary(0.6, 1.0) == pytest.approx(0.2)
    assert kelly_binary(0.4, 1.0) == pytest.approx(-0.2)
    assert kelly_binary(0.5, 0.0) == 0.0
    r = np.random.default_rng(1).normal(0.001, 0.01, 5000)
    assert kelly_continuous(r) == pytest.approx(r.mean() / r.var(ddof=1))


def test_trade_stats_and_sizer():
    s = TradeStats(50)
    for p in [0.02, -0.01] * 20:
        s.add(p)
    assert s.kelly() == pytest.approx(0.5 - 0.5 / 2)          # p=.5, b=2 → .25
    sizer = KellySizer(0.5, 30, 0.25, 0.1, 1.0)
    cap, mode = sizer.cap(s)
    assert mode == "kelly" and cap == pytest.approx(0.125)
    assert sizer.cap(TradeStats())[1] == "bootstrap"
    bad = TradeStats()
    for p in [-0.01, 0.005] * 20:
        bad.add(p)
    assert sizer.cap(bad) == (0.1, "no_edge")                  # nunca 0: permite recuperarse
    only_wins = TradeStats()
    for _ in range(40):
        only_wins.add(0.01)
    assert sizer.cap(only_wins)[1] == "bootstrap"             # sin pérdidas, Kelly no fiable


def test_consecutive_losses():
    s = TradeStats()
    for p in [0.1, -0.1, -0.1, -0.1]:
        s.add(p)
    assert s.consecutive_losses == 3


# ───────────────────────────── VaR
def test_var_historical_and_scaling():
    losses = np.linspace(-0.05, 0.05, 1001)
    assert historical_var(losses, 0.95) == pytest.approx(0.045, abs=1e-3)
    m = VaRModel(0.95, 4, 1000, 100)
    r = rets(1000)
    v1 = VaRModel(0.95, 1, 1000, 100).compute(r, 1, 0.01)
    v4 = m.compute(r, 1, 0.01)
    assert v4.var == pytest.approx(2 * v1.var) and v4.cvar >= v4.var and v4.method == "empirical"


def test_var_fallback_and_side_asymmetry():
    assert VaRModel(0.95, 1, 500, 100).compute(rets(10), 1, 0.01).method == "atr_fallback"
    # serie con caídas bruscas: el largo debe ver más riesgo que el corto
    r = np.where(np.arange(1000) % 50 == 0, -0.08, 0.001)
    m = VaRModel(0.99, 1, 1000, 100)
    assert m.compute(r, 1, 0.01).var > m.compute(r, -1, 0.01).var


def test_cornish_fisher_penalizes_fat_tails():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 0.01, 20000)
    fat = rng.standard_t(3, 20000) * 0.01 / np.sqrt(3)
    assert cornish_fisher_var(-fat, 0.99) > cornish_fisher_var(-normal, 0.99)
    assert max_weight_for_var(0.03, 0.06) == pytest.approx(0.5)
    assert max_weight_for_var(0.03, 0.0) == float("inf")


# ───────────────────────────── stops
def test_long_stop_loss_take_profit_and_conflict():
    p = PositionStops(1, 100, 0.01, 2.0, 4.0, 0)
    assert p.stop == pytest.approx(98) and p.take_profit == pytest.approx(104)
    assert p.update(101, 99).kind if p.update(101, 99) else True
    assert PositionStops(1, 100, 0.01, 2, 4, 0).update(101, 97.9).kind == "stop_loss"
    assert PositionStops(1, 100, 0.01, 2, 4, 0).update(104.1, 99).kind is not None
    both = PositionStops(1, 100, 0.01, 2, 4, 0).update(105, 97)
    assert both.kind == "stop_loss"                            # conservador: SL antes que TP


def test_trailing_only_ratchets():
    p = PositionStops(1, 100, 0.01, 2.0, 0, 1.0)               # trailing a 1 ATR (1.0)
    assert p.update(103, 101) is None and p.stop == pytest.approx(102)
    assert p.update(102.5, 102.2) is None and p.stop == pytest.approx(102)   # no retrocede
    t = p.update(102, 101.9)
    assert t is not None and t.kind == "trailing_stop"


def test_short_stops():
    p = PositionStops(-1, 100, 0.01, 2.0, 4.0, 0)
    assert p.stop == pytest.approx(102) and p.take_profit == pytest.approx(96)
    assert p.update(102.1, 99).kind == "stop_loss"
    assert PositionStops(-1, 100, 0.01, 2, 4, 0).update(101, 95.9).kind == "take_profit"


# ───────────────────────────── breaker
def test_daily_halt_trips_and_releases_next_day(cfg):
    b = CircuitBreaker(cfg)
    assert b.update(10_000, T0) is BreakerState.OK
    assert b.update(9_650, T0 + timedelta(hours=1)) is BreakerState.DAILY_HALT
    assert b.update(9_650, T0 + timedelta(hours=5)) is BreakerState.DAILY_HALT
    assert b.update(9_650, T0 + timedelta(days=1)) is BreakerState.OK


def test_drawdown_halt_manual_reset_and_cooldown(cfg):
    b = CircuitBreaker(cfg)
    b.update(10_000, T0)
    for i in range(1, 5):                                      # baja gradual en días distintos
        b.update(10_000 - i * 400, T0 + timedelta(days=i))
    assert b.state is BreakerState.DRAWDOWN_HALT
    b.update(8_000, T0 + timedelta(days=30))
    assert b.state is BreakerState.DRAWDOWN_HALT               # cooldown=0 → solo manual
    b.reset(8_000)
    assert b.update(8_000, T0 + timedelta(days=31)) is BreakerState.OK
    b2 = CircuitBreaker(replace(cfg, drawdown_cooldown_hours=24))
    b2.update(10_000, T0)
    b2.update(8_000, T0 + timedelta(days=1))
    assert b2.state is BreakerState.DRAWDOWN_HALT
    assert b2.update(8_000, T0 + timedelta(days=1, hours=25)) is BreakerState.OK


def test_breaker_persists_across_restart(cfg):
    path = cfg.state_dir / "b.json"
    b = CircuitBreaker(cfg, path)
    b.update(10_000, T0)
    b.update(9_000, T0 + timedelta(hours=1))
    assert path.exists()
    b2 = CircuitBreaker(cfg, path)
    assert b2.state is not BreakerState.OK and b2.peak_equity == 10_000


def test_breaker_caps_configured_daily_loss_at_three_percent(cfg):
    b = CircuitBreaker(replace(cfg, max_daily_loss_pct=0.05))
    b.update(10_000, T0)
    assert b.update(9_700, T0 + timedelta(hours=1)) is BreakerState.DAILY_HALT


def test_daily_loss_halt_overrides_streak_cooldown(cfg):
    b = CircuitBreaker(cfg)
    b.update(10_000, T0)
    b.register_trade(cfg.max_consecutive_losses, T0)

    assert b.update(9_700, T0 + timedelta(minutes=1)) is BreakerState.DAILY_HALT


def test_invalid_equity_fails_closed(cfg):
    b = CircuitBreaker(cfg)
    b.update(10_000, T0)
    assert b.update(float("nan"), T0 + timedelta(hours=1)) is BreakerState.DRAWDOWN_HALT


def test_corrupt_breaker_state_fails_closed(cfg):
    path = cfg.state_dir / "breaker.json"
    path.write_text("{invalid", encoding="utf-8")

    breaker = CircuitBreaker(cfg, path)

    assert breaker.state is BreakerState.DRAWDOWN_HALT
    assert json.loads(path.read_text())["state"] == BreakerState.DRAWDOWN_HALT.value


def test_loss_streak_cooldown(cfg):
    b = CircuitBreaker(cfg)
    b.update(10_000, T0)
    b.register_trade(5, T0)
    assert b.state is BreakerState.STREAK_COOLDOWN
    assert b.update(10_000, T0 + timedelta(minutes=30)) is BreakerState.STREAK_COOLDOWN
    assert b.update(10_000, T0 + timedelta(minutes=121)) is BreakerState.OK


# ───────────────────────────── manager
def test_allows_reasonable_request(cfg):
    rm = mk(cfg)
    d = rm.evaluate(state(), mkt(), 0.2, rets())
    assert d.action is RiskAction.ALLOW and d.approved_weight == pytest.approx(0.2)
    assert d.stop_price == pytest.approx(98.0) and d.take_profit_price == pytest.approx(104.0)


def test_current_stops_returns_snapshot_of_active_protection(cfg):
    rm = mk(cfg)
    rm.on_position_opened("BTC/USDT", 1, 100.0, 0.01)

    snapshot = rm.current_stops()

    assert snapshot["BTC/USDT"] == {
        "side": 1,
        "entry": 100.0,
        "stop": pytest.approx(98.0),
        "initial_stop": pytest.approx(98.0),
        "take_profit": pytest.approx(104.0),
        "extreme": 100.0,
    }
    snapshot.clear()
    assert "BTC/USDT" in rm.current_stops()


def test_risk_manager_requires_exchange_reconciled_state_by_default(cfg):
    rm = RiskManager(cfg, persist=False)
    with pytest.raises(ValueError, match="conciliado con el exchange"):
        rm.evaluate(state(), mkt(), 0.2, rets())

    exchange_state = replace(state(), source=PortfolioSource.EXCHANGE)
    decision = rm.evaluate(exchange_state, mkt(), 0.2, rets())
    assert decision.approved_weight == pytest.approx(0.2)


def test_spot_blocks_shorts(cfg):
    d = mk(cfg).evaluate(state(), mkt(), -0.5, rets())
    assert d.approved_weight == 0.0 and "NO_SHORT" in d.codes


def test_max_exposure_and_headroom(cfg):
    rm = mk(replace(cfg, kelly_bootstrap_cap=1.0, var_limit_pct=1.0))
    first = rm.evaluate(state(), mkt(), 1.0, rets())
    assert first.approved_weight == pytest.approx(0.25) and "HARD_ORDER_LIMIT" in first.codes
    d = rm.evaluate(state(w=0.2, others=0.7), mkt(), 1.0, rets())
    assert d.approved_weight == pytest.approx(0.3) and "MAX_EXPOSURE" in d.codes


def test_hard_order_and_leverage_caps_override_configuration(cfg):
    permissive = replace(
        cfg, max_position_frac=3.0, max_total_exposure=3.0, max_leverage=3.0,
        kelly_bootstrap_cap=3.0, var_limit_pct=1.0,
    )
    decision = mk(permissive).evaluate(state(), mkt(), 3.0, rets())

    assert decision.approved_weight == pytest.approx(0.25)
    assert "HARD_ORDER_LIMIT" in decision.codes
    assert abs(decision.approved_weight) <= 1.0


def test_invalid_exposure_limit_fails_closed(cfg):
    invalid = replace(cfg, max_leverage=float("nan"), kelly_bootstrap_cap=1.0)
    decision = mk(invalid).evaluate(state(), mkt(), 0.2, rets())

    assert decision.approved_weight == 0.0
    assert "INVALID_RISK_LIMIT" in decision.codes


def test_non_finite_request_is_rejected(cfg):
    with pytest.raises(ValueError, match="finitas"):
        mk(cfg).evaluate(state(), mkt(), float("nan"), rets())


def test_stop_loss_cannot_be_disabled_or_widened_past_two_atr(cfg):
    disabled = mk(replace(cfg, sl_atr_mult=0.0, tp_atr_mult=0.0, trail_atr_mult=0.0))
    decision = disabled.evaluate(
        state(w=0.5, entry=100.0), mkt(low=97.9, price=97.9), 0.5, rets()
    )
    assert decision.action is RiskAction.FLATTEN
    assert decision.approved_weight == 0.0

    widened = mk(replace(cfg, sl_atr_mult=20.0, tp_atr_mult=0.0, trail_atr_mult=0.0))
    preview = widened.evaluate(state(), mkt(), 0.2, rets())
    assert preview.stop_price == pytest.approx(98.0)


def test_stop_loss_flatten_bypasses_normal_order_and_exchange_minimums(cfg):
    restrictive = replace(cfg, min_notional=10_000.0, sl_atr_mult=0.0)
    rm = mk(restrictive)
    decision = rm.evaluate(
        state(w=0.5, equity=100.0, entry=100.0),
        mkt(low=97.9, price=97.9),
        0.5,
        rets(),
    )

    assert decision.action is RiskAction.FLATTEN
    assert decision.approved_weight == 0.0


def test_kelly_limits_additions_but_not_existing_position(cfg):
    rm = mk(cfg)                                               # bootstrap cap 0.25
    d = rm.evaluate(state(), mkt(), 0.9, rets())
    assert d.approved_weight == pytest.approx(0.25) and "HARD_ORDER_LIMIT" in d.codes
    held = rm.evaluate(state(w=0.6, entry=100.0), mkt(), 0.6, rets())
    assert held.approved_weight == pytest.approx(0.6)          # no fuerza a vender lo que ya tiene
    grow = rm.evaluate(state(w=0.6, entry=100.0), mkt(), 0.9, rets())
    assert grow.approved_weight == pytest.approx(0.6)


def test_var_limit_reduces_hard_even_for_open_position(cfg):
    rm = mk(replace(cfg, kelly_bootstrap_cap=1.0, var_limit_pct=0.02))
    wild = rets(500, sigma=0.03)
    d = rm.evaluate(state(w=0.8, entry=100.0), mkt(), 0.8, wild)
    assert "VAR_LIMIT" in d.codes and d.approved_weight < 0.8
    assert d.approved_weight * d.var_unit <= 0.02 + 1e-9


def test_stop_loss_flattens_and_blocks_reentry(cfg):
    rm = mk(replace(cfg, kelly_bootstrap_cap=1.0, var_limit_pct=1.0))
    rm.on_position_opened("BTC/USDT", 1, 100.0, 0.01)
    d = rm.evaluate(state(w=0.5, entry=100.0), mkt(low=97.0, price=97.5), 0.5, rets())
    assert d.action is RiskAction.FLATTEN and d.approved_weight == 0.0 and "STOP_LOSS" in d.codes
    rm.on_position_closed("BTC/USDT", 1, 100.0, 97.9, T0, reason="stop_loss")
    d2 = rm.evaluate(state(t=T0 + timedelta(minutes=10)), mkt(), 0.5, rets())
    assert d2.approved_weight == 0.0 and "REENTRY_COOLDOWN" in d2.codes
    d3 = rm.evaluate(state(t=T0 + timedelta(minutes=90)), mkt(), 0.5, rets())
    assert d3.approved_weight > 0


def test_stop_levels_recovered_after_restart(cfg):
    rm = mk(cfg)                                               # no hay tracker: se reconstruye
    d = rm.evaluate(state(w=0.3, entry=100.0), mkt(low=97.0), 0.3, rets())
    assert d.action is RiskAction.FLATTEN


def test_breaker_drawdown_flattens_and_blocks(cfg):
    rm = mk(replace(cfg, kelly_bootstrap_cap=1.0, var_limit_pct=1.0))
    rm.evaluate(state(equity=10_000), mkt(), 0.0, rets())
    d = rm.evaluate(state(w=0.5, equity=8_000, entry=100.0, t=T0 + timedelta(days=2)), mkt(), 0.5, rets())
    assert d.breaker is BreakerState.DRAWDOWN_HALT and d.action is RiskAction.FLATTEN
    d2 = rm.evaluate(state(w=0.0, equity=8_000, t=T0 + timedelta(days=3)), mkt(), 0.8, rets())
    assert d2.approved_weight == 0.0


def test_daily_halt_forces_flatten_even_if_configuration_disables_it(cfg):
    rm = mk(replace(cfg, kelly_bootstrap_cap=1.0, var_limit_pct=1.0))
    rm.evaluate(state(equity=10_000), mkt(), 0.0, rets())
    t = T0 + timedelta(hours=2)
    up = rm.evaluate(state(w=0.4, equity=9_600, entry=100.0, t=t), mkt(), 0.9, rets())
    assert up.breaker is BreakerState.DAILY_HALT
    assert up.action is RiskAction.FLATTEN and up.approved_weight == 0.0
    assert "BREAKER_DAILY" in up.codes
    down = rm.evaluate(state(equity=9_600, t=t), mkt(), 0.9, rets())
    assert down.approved_weight == 0.0


def test_sentiment_overlay(cfg):
    rm = mk(replace(cfg, kelly_bootstrap_cap=1.0, var_limit_pct=1.0))
    base = rm.evaluate(state(), mkt(), 0.6, rets(), None).approved_weight
    bear = rm.evaluate(state(), mkt(), 0.6, rets(), Sent(20)).approved_weight
    assert bear == pytest.approx(base * cfg.sentiment_derisk_factor)
    assert rm.evaluate(state(), mkt(), 0.6, rets(), Sent(20, degraded=True)).approved_weight == base
    assert rm.evaluate(state(), mkt(), 0.6, rets(), Sent(20, confidence=0.1)).approved_weight == base
    assert rm.evaluate(state(), mkt(), 0.6, rets(), Sent(90)).approved_weight == base   # nunca añade riesgo


def test_spread_participation_and_min_notional(cfg):
    rm = mk(replace(cfg, kelly_bootstrap_cap=1.0, var_limit_pct=1.0))
    assert rm.evaluate(state(), mkt(spread_bps=50), 0.5, rets()).approved_weight == 0.0
    d = rm.evaluate(state(), mkt(bar_quote_volume=100_000), 0.5, rets())   # máx 2% = 2000 USDT = 0.2
    assert d.approved_weight == pytest.approx(0.2) and "PARTICIPATION" in d.codes
    small = mk(replace(cfg, kelly_bootstrap_cap=1.0, var_limit_pct=1.0)).evaluate(state(equity=100.0), mkt(), 0.05, rets())          # 5 USDT < 10
    assert small.action is RiskAction.HOLD and small.approved_weight == 0.0


def test_reducing_is_never_blocked_by_entry_rules(cfg):
    rm = mk(replace(cfg, kelly_bootstrap_cap=0.1))
    d = rm.evaluate(state(w=0.5, entry=100.0), mkt(spread_bps=500), 0.0, rets())
    assert d.approved_weight == pytest.approx(0.25) and d.action is RiskAction.REDUCE


def test_futures_allow_short_with_var_on_upper_tail(tmp_path):
    c = replace(RiskSettings.from_env(MarketType.FUTURE, "1h"), state_dir=tmp_path,
                allow_short=True, kelly_bootstrap_cap=1.0, var_limit_pct=1.0)
    d = RiskManager(c, persist=False, require_exchange_state=False).evaluate(
        state(), mkt(), -0.4, rets()
    )
    assert d.approved_weight == pytest.approx(-0.25)
    assert d.stop_price == pytest.approx(102.0)


def test_events_audit_and_persistence(cfg):
    bus, got = EventBus(), []
    bus.subscribe(RiskEvent, got.append)
    rm = RiskManager(cfg, bus=bus, persist=True, require_exchange_state=False)
    d = asyncio.run(rm.evaluate_and_publish(state(), mkt(), 0.9, rets()))
    assert got and got[0].approved_weight == d.approved_weight and "HARD_ORDER_LIMIT" in got[0].reasons
    lines = (cfg.state_dir / "decisions.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["action"] == "REDUCE"
    rm.on_position_closed("BTC/USDT", 1, 100, 103, T0, fees_pct=0.002)
    rm2 = RiskManager(cfg, persist=True, require_exchange_state=False)
    assert len(rm2.stats) == 1 and rm2.stats.to_list()[0] == pytest.approx(0.028)


def test_bars_per_day():
    assert bars_per_day("1h") == 24 and bars_per_day("15m") == 96
