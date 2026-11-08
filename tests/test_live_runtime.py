import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.events import KlineEvent, SentimentEvent
from config.live_settings import ExecutionSettings
from config.rl_settings import CostSettings
from src.execution.paper import OrderExecutor, PaperGateway
from src.live.orchestrator import (
    StateStore,
    TradingOrchestrator,
    find_model_dir,
    target_weight_from_prediction,
)
from src.risk.models import RiskAction


def test_paper_gateway_simulates_buy_and_flatten_without_network():
    gateway = PaperGateway(
        lambda _: (100.0, 1_000_000.0),
        initial_cash=1_000.0,
        costs=CostSettings(taker_fee_bps=0, base_slippage_bps=0, impact_coef_bps=0),
    )

    fill = gateway.rebalance("BTC/USDT", 0.5, 1_000.0, 100.0, 1_000_000.0)
    assert fill["side"] == "buy"
    assert gateway.qty("BTC/USDT") == pytest.approx(5.0)
    assert gateway.equity({"BTC/USDT": 100.0}) == pytest.approx(1_000.0)

    gateway.rebalance("BTC/USDT", 0.0, 1_000.0, 100.0, 1_000_000.0)
    assert gateway.qty("BTC/USDT") == pytest.approx(0.0)
    assert gateway.cash == pytest.approx(1_000.0)


def test_paper_spot_rejects_opening_short():
    gateway = PaperGateway(lambda _: (100.0, 1_000_000.0), 1_000.0, CostSettings())

    with pytest.raises(ValueError, match="no permite abrir posiciones cortas"):
        gateway.rebalance("BTC/USDT", -0.2, 1_000.0, 100.0, 1_000_000.0)


def test_paper_executor_allows_risk_reduction_below_minimum_threshold():
    gateway = PaperGateway(
        lambda _: (100.0, 1_000_000.0),
        1_000.0,
        CostSettings(taker_fee_bps=0, base_slippage_bps=0, impact_coef_bps=0),
    )
    gateway.rebalance("BTC/USDT", 0.10, 1_000.0, 100.0, 1_000_000.0)
    executor = OrderExecutor(gateway, ExecutionSettings(min_rebalance=0.05))

    fill = asyncio.run(executor.execute("BTC/USDT", 0.08, 1_000.0, 100.0, 1_000_000.0))

    assert fill is not None
    assert gateway.qty("BTC/USDT") == pytest.approx(0.8)


def test_state_store_round_trips_json(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.save({"mode": "PAPER", "equity": 123.4})

    assert store.load() == {"mode": "PAPER", "equity": 123.4}


def test_find_model_dir_reports_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="Entrena un modelo"):
        find_model_dir("BTC/USDT", "1h", tmp_path)


@pytest.mark.parametrize(
    ("action", "allow_short", "expected"),
    [(-0.25, False, 0.0), (0.75, False, 0.75), (-0.25, True, -0.25)],
)
def test_live_prediction_respects_training_market_exposure_range(action, allow_short, expected):
    rl = SimpleNamespace(max_leverage=1.0, allow_short=allow_short)

    assert target_weight_from_prediction(action, rl) == pytest.approx(expected)


def test_sentiment_status_is_explanatory_before_first_result():
    orchestrator = TradingOrchestrator.__new__(TradingOrchestrator)
    orchestrator.symbols = ["BTC/USDT"]
    orchestrator._sentiment = {}
    orchestrator.sentiment_enabled = True

    status = orchestrator._sentiment_snapshot()["BTC"]

    assert status["status"] == "PENDING"
    assert status["score"] is None
    assert status["reason"]


def test_sentiment_status_contains_latest_published_result():
    orchestrator = TradingOrchestrator.__new__(TradingOrchestrator)
    orchestrator.symbols = ["BTC/USDT"]
    orchestrator.sentiment_enabled = True
    orchestrator._sentiment = {
        "BTC": SentimentEvent(
            asset="BTC",
            score=72.0,
            confidence=0.8,
            n_scored=4,
            fear_greed=55,
            degraded=False,
            timestamp=1_791_740_000_000,
        )
    }

    status = orchestrator._sentiment_snapshot()["BTC"]

    assert status["status"] == "ACTIVE"
    assert status["score"] == 72.0
    assert status["confidence"] == 0.8
    assert status["age_min"] >= 0


def test_manual_proposal_does_not_execute_until_approved():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    event = KlineEvent(
        symbol="BTC/USDT", timeframe="1m", open_time=now_ms,
        open=100.0, high=101.0, low=99.0, close=100.0, volume=1_000.0,
        is_closed=True,
    )

    class Buffer:
        def candles(self, symbol):
            assert symbol == "BTC/USDT"
            return pd.DataFrame(
                [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000.0}],
                index=[pd.Timestamp(now_ms, unit="ms", tz="UTC")],
            )

        def order_book(self, symbol):
            assert symbol == "BTC/USDT"
            return None

    class Risk:
        def __init__(self):
            self.calls = []
            self.approved_weight = 0.5

        async def evaluate_and_publish(self, *args):
            self.calls.append(args)
            return SimpleNamespace(
                approved_weight=self.approved_weight,
                action=RiskAction.ALLOW,
                codes=(),
                reasons=(),
            )

        def on_position_opened(self, symbol, side, entry_price, atr_pct):
            assert symbol == "BTC/USDT"
            assert side == 1 and entry_price > 0 and atr_pct > 0

    gateway = PaperGateway(
        lambda _: (100.0, 100_000.0),
        1_000.0,
        CostSettings(taker_fee_bps=0, base_slippage_bps=0, impact_coef_bps=0),
    )
    orchestrator = TradingOrchestrator.__new__(TradingOrchestrator)
    orchestrator.symbols = ["BTC/USDT"]
    orchestrator.buffer = Buffer()
    orchestrator.gateway = gateway
    orchestrator.executor = OrderExecutor(gateway, ExecutionSettings(min_rebalance=0.05))
    orchestrator.risk = Risk()
    orchestrator.mode = "PAPER"
    orchestrator.manual_approval = True
    orchestrator._pending_proposals = {}
    orchestrator._proposal_context = {}
    orchestrator._status_sym = {"BTC/USDT": {}}
    orchestrator._save = lambda: None
    orchestrator._write_status = lambda: None
    orchestrator.trade_log = SimpleNamespace(fill=lambda _: None)

    decision = SimpleNamespace(
        approved_weight=0.5,
        action=RiskAction.ALLOW,
        codes=(),
        reasons=(),
    )
    proposal = orchestrator._create_proposal(event, 100.0, 1_000.0, 0.0, 0.8, decision)
    orchestrator._proposal_context[proposal["id"]] = {
        "event": event,
        "returns": [],
        "requested": 0.8,
        "entries_paused": False,
        "sentiment": None,
        "atr_pct": 0.02,
    }
    assert gateway.qty("BTC/USDT") == 0.0

    orchestrator.risk.approved_weight = 0.4
    with pytest.raises(ValueError, match="El riesgo cambió"):
        asyncio.run(orchestrator.handle_manual_approval(proposal["id"], True))
    assert gateway.qty("BTC/USDT") == 0.0

    orchestrator.risk.approved_weight = 0.5
    proposal = orchestrator._create_proposal(event, 100.0, 1_000.0, 0.0, 0.8, decision)
    orchestrator._proposal_context[proposal["id"]] = {
        "event": event,
        "returns": [],
        "requested": 0.8,
        "entries_paused": False,
        "sentiment": None,
        "atr_pct": 0.02,
    }
    result = asyncio.run(orchestrator.handle_manual_approval(proposal["id"], True))

    assert result["status"] == "executed"
    assert gateway.qty("BTC/USDT") == pytest.approx(5.0)
    assert len(orchestrator.risk.calls) == 2
