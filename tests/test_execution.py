import asyncio
import hashlib
import hmac
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import BinanceEnv, MarketType, Settings
from src.core.events import EventBus, ExecutionReportEvent
from src.execution.ledger import ExecutionLedger, OrderStatusLedger
from src.execution.portfolio import ExchangePortfolioReconciler, PortfolioReconciliationError
from src.execution.user_stream import BinanceUserDataStream, parse_execution_report
from src.risk.models import PortfolioSource


def settings(tmp_path, market_type=MarketType.SPOT, symbols=("BTC/USDT",)):
    return Settings(
        env=BinanceEnv.TESTNET,
        market_type=market_type,
        api_key="read-only-key",
        api_secret="read-only-secret",
        symbols=symbols,
        timeframe="1h",
        history_days=30,
        historical_from_live=False,
        ws_depth_levels=20,
        log_level="INFO",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )


def test_parse_spot_execution_report_preserves_each_partial_fill():
    event = parse_execution_report({
        "e": "executionReport",
        "s": "BTCUSDT",
        "c": "client-1",
        "i": 77,
        "x": "TRADE",
        "X": "PARTIALLY_FILLED",
        "S": "BUY",
        "l": "0.01",
        "L": "65000.25",
        "z": "0.01",
        "Z": "650.0025",
        "n": "0.00001",
        "N": "BTC",
        "t": 991,
        "T": 1_760_000_000_000,
    }, MarketType.SPOT, lambda market_id: "BTC/USDT")

    assert event is not None and event.is_fill
    assert event.symbol == "BTC/USDT"
    assert event.last_filled_quantity == pytest.approx(0.01)
    assert event.last_filled_price == pytest.approx(65000.25)
    assert event.commission == pytest.approx(0.00001)
    assert event.commission_asset == "BTC"
    assert event.order_status == "PARTIALLY_FILLED"


def test_parse_futures_order_update_and_ignore_account_events():
    event = parse_execution_report({
        "e": "ORDER_TRADE_UPDATE",
        "E": 1_760_000_000_000,
        "o": {
            "s": "BTCUSDT",
            "c": "client-2",
            "i": 88,
            "x": "TRADE",
            "X": "FILLED",
            "S": "SELL",
            "l": "0.02",
            "L": "64000",
            "z": "0.02",
            "Z": "1280",
            "n": "0.64",
            "N": "USDT",
            "t": 992,
            "T": 1_760_000_000_001,
            "rp": "1.25",
        },
    }, MarketType.FUTURE)

    assert event is not None and event.is_fill
    assert event.realized_pnl == pytest.approx(1.25)
    assert parse_execution_report({"e": "ACCOUNT_UPDATE"}, MarketType.FUTURE) is None


def test_ledger_deduplicates_exchange_trade_ids_across_restart(tmp_path):
    event = parse_execution_report({
        "e": "executionReport", "s": "BTCUSDT", "i": 17, "x": "TRADE",
        "X": "PARTIALLY_FILLED", "S": "BUY", "l": "0.1", "L": "100",
        "z": "0.1", "Z": "10", "n": "0.01", "N": "USDT", "t": 5,
    }, MarketType.SPOT, lambda _: "BTC/USDT")
    assert event is not None
    path = tmp_path / "execution" / "fills.jsonl"

    assert ExecutionLedger(path).record(event)
    assert not ExecutionLedger(path).record(event)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["last_filled_price"] == 100.0
    assert record["commission_asset"] == "USDT"


def test_order_status_ledger_persists_and_deduplicates_rejections(tmp_path):
    rejected = parse_execution_report({
        "e": "executionReport", "s": "BTCUSDT", "i": 18, "x": "REJECTED",
        "X": "REJECTED", "S": "BUY",
    }, MarketType.SPOT, lambda _: "BTC/USDT")
    assert rejected is not None and not rejected.is_fill
    path = tmp_path / "orders.jsonl"

    assert OrderStatusLedger(path).record(rejected)
    assert not OrderStatusLedger(path).record(rejected)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["execution_type"] == "REJECTED"
    assert record["order_status"] == "REJECTED"


def test_spot_user_stream_uses_signed_websocket_api_subscription(tmp_path):
    class FakeExchange:
        markets = {}

    cfg = settings(tmp_path)
    stream = BinanceUserDataStream(
        cfg, FakeExchange(), EventBus(), ExecutionLedger(tmp_path / "fills.jsonl")
    )

    request = stream._spot_subscription_request("test-request")
    params = request["params"]
    signature_payload = f"apiKey={params['apiKey']}&timestamp={params['timestamp']}"
    expected_signature = hmac.new(
        cfg.api_secret.encode("utf-8"),
        signature_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert request["method"] == "userDataStream.subscribe.signature"
    assert params["signature"] == expected_signature

    class FakeWebSocket:
        sent = None

        async def send(self, message):
            self.sent = json.loads(message)

        async def recv(self):
            return json.dumps({
                "id": self.sent["id"],
                "status": 200,
                "result": {"subscriptionId": 3},
            })

    asyncio.run(stream._subscribe_spot(FakeWebSocket()))

    calls = []

    class FakeFutureExchange:
        markets = {}

        async def fapiPrivatePostListenKey(self):
            return {"listenKey": "future-token"}

        async def fapiPrivatePutListenKey(self, params):
            calls.append(("future-renew", params["listenKey"]))

        async def fapiPrivateDeleteListenKey(self, params):
            calls.append(("future-delete", params["listenKey"]))

    future = BinanceUserDataStream(
        settings(tmp_path, MarketType.FUTURE), FakeFutureExchange(),
        EventBus(), ExecutionLedger(tmp_path / "future-fills.jsonl"),
    )
    assert asyncio.run(future._create_listen_key()) == "future-token"
    asyncio.run(future._keepalive("future-token"))
    assert calls[-1] == ("future-renew", "future-token")
    asyncio.run(future._delete_listen_key("future-token"))
    assert calls[-1] == ("future-delete", "future-token")


def test_user_stream_persists_fill_and_publishes_order_event(tmp_path):
    class FakeExchange:
        markets = {}

    cfg = settings(tmp_path)
    bus = EventBus()
    received = []
    bus.subscribe(ExecutionReportEvent, received.append)
    ledger_path = tmp_path / "fills.jsonl"
    orders_path = tmp_path / "orders.jsonl"
    stream = BinanceUserDataStream(
        cfg, FakeExchange(), bus, ExecutionLedger(ledger_path),
        OrderStatusLedger(orders_path),
    )

    asyncio.run(stream._dispatch(json.dumps({
        "subscriptionId": 3,
        "event": {
            "e": "executionReport", "s": "BTCUSDT", "i": 77, "x": "TRADE",
            "X": "PARTIALLY_FILLED", "S": "BUY", "l": "0.01", "L": "65000",
            "z": "0.01", "Z": "650", "n": "0.02", "N": "USDT", "t": 21,
        },
    })))

    assert len(received) == 1 and received[0].last_filled_price == 65000
    assert json.loads(ledger_path.read_text(encoding="utf-8"))["commission"] == 0.02
    assert json.loads(orders_path.read_text(encoding="utf-8"))["order_status"] == "PARTIALLY_FILLED"


class FakeSpotExchange:
    markets = {"BTC/USDT": {}, "ETH/USDT": {}}

    async def load_markets(self):
        return self.markets

    async def fetch_balance(self):
        return {"total": {"USDT": 1_000, "BTC": 0.1, "ETH": 0.2}}

    async def fetch_ticker(self, symbol):
        return {"last": {"BTC/USDT": 50_000, "ETH/USDT": 2_500}[symbol]}


def test_spot_portfolio_state_uses_account_balances_and_exchange_prices(tmp_path):
    cfg = settings(tmp_path, symbols=("BTC/USDT", "ETH/USDT"))
    states = asyncio.run(ExchangePortfolioReconciler(FakeSpotExchange(), cfg).fetch_states())

    btc = states["BTC/USDT"]
    assert btc.source is PortfolioSource.EXCHANGE
    assert btc.equity == pytest.approx(6_500)
    assert btc.price == 50_000
    assert btc.weight == pytest.approx(5_000 / 6_500)
    assert btc.other_exposure == pytest.approx(500 / 6_500)


def test_spot_reconciliation_fails_closed_for_unpriced_assets(tmp_path):
    class UnknownAssetExchange(FakeSpotExchange):
        async def fetch_balance(self):
            return {"total": {"USDT": 1_000, "UNKNOWN": 1}}

    with pytest.raises(PortfolioReconciliationError, match="No se puede conciliar"):
        asyncio.run(ExchangePortfolioReconciler(
            UnknownAssetExchange(), settings(tmp_path)
        ).fetch_states())


def test_futures_portfolio_state_uses_exchange_positions(tmp_path):
    symbols = ("BTC/USDT:USDT", "ETH/USDT:USDT")

    class FakeFutureExchange:
        markets = {
            symbol: {
                "symbol": symbol,
                "quote": "USDT",
                "contractSize": 1.0,
            } for symbol in symbols
        }

        async def load_markets(self):
            return self.markets

        async def fetch_balance(self):
            return {"total": {"USDT": 5_000}}

        async def fetch_positions(self, requested=()):
            assert requested == ()
            return [
                {
                    "symbol": "BTC/USDT:USDT",
                    "contracts": 0.1,
                    "side": "long",
                    "entryPrice": 48_000,
                    "markPrice": 50_000,
                },
                {
                    "symbol": "ETH/USDT:USDT",
                    "contracts": 0.1,
                    "side": "short",
                    "entryPrice": 2_600,
                    "markPrice": 2_500,
                },
            ]

        async def fetch_ticker(self, symbol):
            return {"last": 50_000 if symbol == symbols[0] else 2_500}

        def market(self, symbol):
            if symbol == "BTC/USDT":
                return self.markets[symbols[0]]
            if symbol == "ETH/USDT":
                return self.markets[symbols[1]]
            return self.markets[symbol]

    states = asyncio.run(ExchangePortfolioReconciler(
        FakeFutureExchange(), settings(tmp_path, MarketType.FUTURE, symbols)
    ).fetch_states())

    assert states[symbols[0]].source is PortfolioSource.EXCHANGE
    assert states[symbols[0]].equity == 5_000
    assert states[symbols[0]].weight == pytest.approx(1.0)
    assert states[symbols[0]].entry_price == 48_000
    assert states[symbols[0]].other_exposure == pytest.approx(0.05)
    assert states[symbols[1]].weight == pytest.approx(-0.05)


def test_reconciler_requires_read_credentials(tmp_path):
    cfg = replace(settings(tmp_path), api_key="", api_secret="")
    with pytest.raises(ValueError, match="credenciales"):
        ExchangePortfolioReconciler(FakeSpotExchange(), cfg)
