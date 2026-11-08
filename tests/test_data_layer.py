import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from src.core.events import EventBus, KlineEvent, OrderBookEvent
from src.data.buffer import MarketDataBuffer
from src.data.exchange_factory import ExchangeFactory
from src.data.features import FeatureEngineer
from src.data.historical import HistoricalDataDownloader
from src.data.streams import BinanceWebSocketClient


def _ohlcv(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)), "high": close * 1.005,
        "low": close * 0.995, "close": close, "volume": rng.uniform(10, 100, n),
    }, index=idx)


# ── Features ────────────────────────────────────────────────────────────────
def test_features_columnas_sin_nans():
    fe = FeatureEngineer()
    out = fe.transform(_ohlcv())
    assert set(fe.FEATURE_COLUMNS) <= set(out.columns)
    assert out[fe.FEATURE_COLUMNS].isna().sum().sum() == 0
    assert len(out) > 0


def test_features_sin_lookahead():
    """Los features en t no cambian si se recortan los datos futuros."""
    fe, df = FeatureEngineer(), _ohlcv()
    full, cut = fe.transform(df), fe.transform(df.iloc[:400])
    diff = (full.loc[cut.index, fe.FEATURE_COLUMNS] - cut[fe.FEATURE_COLUMNS]).abs().max().max()
    assert diff < 1e-9


# ── WebSocket: parseo de mensajes ───────────────────────────────────────────
def _client(symbols=("BTC/USDT",)):
    settings = SimpleNamespace(ws_base_url="wss://x", ws_depth_levels=20, symbols=symbols, timeframe="1m")
    bus = EventBus()
    events = []
    bus.subscribe(KlineEvent, events.append)
    bus.subscribe(OrderBookEvent, events.append)
    return BinanceWebSocketClient(settings, bus, symbols=list(symbols), timeframe="1m"), events


def test_url_streams_combinados():
    c, _ = _client(("BTC/USDT", "ETH/USDT"))
    url = c.build_url()
    assert url.startswith("wss://x/stream?streams=")
    for s in ("btcusdt@kline_1m", "btcusdt@depth20@100ms", "ethusdt@kline_1m"):
        assert s in url


def test_dispatch_kline_y_depth_spot_y_futures():
    c, events = _client()
    kline = {"stream": "btcusdt@kline_1m", "data": {"e": "kline", "k": {
        "t": 1700000000000, "i": "1m", "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10", "x": True}}}
    spot = {"stream": "btcusdt@depth20@100ms", "data": {
        "lastUpdateId": 1, "bids": [["100", "1"], ["99", "2"]], "asks": [["101", "1"], ["102", "3"]]}}
    fut = {"stream": "btcusdt@depth20@100ms", "data": {
        "e": "depthUpdate", "b": [["100", "1"]], "a": [["101", "1"]]}}

    async def go():
        for m in (kline, spot, fut, {"stream": "otro@kline_1m", "data": {}}):
            await c._dispatch(json.dumps(m))
    asyncio.run(go())

    k, b1, b2 = events
    assert isinstance(k, KlineEvent) and k.is_closed and k.symbol == "BTC/USDT" and k.close == 1.5
    assert b1.best_bid == 100 and b1.best_ask == 101
    assert b1.imbalance() == pytest.approx((3 - 4) / 7)
    assert b2.best_bid == 100  # payload de futuros (claves b/a)


# ── Buffer ──────────────────────────────────────────────────────────────────
def test_buffer_ignora_abiertas_y_reemplaza_duplicados():
    bus = EventBus()
    buf = MarketDataBuffer(bus)

    def ev(t, close, closed):
        return KlineEvent("BTC/USDT", "1m", t, 1, 2, 0.5, close, 10, closed)

    async def go():
        await bus.publish(ev(60_000, 1.0, False))   # abierta: se ignora
        await bus.publish(ev(60_000, 1.1, True))
        await bus.publish(ev(60_000, 1.2, True))    # duplicado (reconexión): reemplaza
        await bus.publish(ev(120_000, 1.3, True))
    asyncio.run(go())

    df = buf.candles("BTC/USDT")
    assert list(df["close"]) == [1.2, 1.3]


def test_buffer_upsert_replaces_timestamp_and_keeps_candles_ordered():
    buffer = MarketDataBuffer(EventBus(), max_len=3)
    idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")

    for timestamp, close in zip(idx, (10.0, 11.0, 12.0)):
        buffer.upsert("BTC/USDT", pd.Series(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1.0},
            name=timestamp,
        ))
    buffer.upsert("BTC/USDT", pd.Series(
        {"open": 9.0, "high": 13.0, "low": 8.0, "close": 12.5, "volume": 2.0},
        name=idx[1],
    ))

    candles = buffer.candles("BTC/USDT")
    assert candles.index.tolist() == list(idx)
    assert candles["close"].tolist() == [10.0, 12.5, 12.0]


def test_buffer_upsert_rejects_row_without_timestamp():
    buffer = MarketDataBuffer(EventBus())
    row = pd.Series({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1})

    with pytest.raises(ValueError, match="timestamp"):
        buffer.upsert("BTC/USDT", row)


# ── Settings / Factory ──────────────────────────────────────────────────────
def test_candado_produccion(monkeypatch):
    monkeypatch.setenv("BINANCE_ENV", "live")
    monkeypatch.setenv("CONFIRM_LIVE_TRADING", "NO")
    with pytest.raises(RuntimeError):
        Settings.from_env()


def test_invalid_ws_depth_uses_safe_default_without_logging_value(monkeypatch, caplog):
    contaminated_value = "2UNRELATED_SECRET_TEXT"
    monkeypatch.setenv("BINANCE_ENV", "testnet")
    monkeypatch.setenv("BINANCE_MARKET_TYPE", "spot")
    monkeypatch.setenv("WS_DEPTH_LEVELS", contaminated_value)

    with caplog.at_level("WARNING"):
        settings = Settings.from_env()

    assert settings.ws_depth_levels == 20
    assert "WS_DEPTH_LEVELS" in caplog.text
    assert "20" in caplog.text
    assert contaminated_value not in caplog.text


def test_factory_testnet_usa_sandbox(monkeypatch):
    monkeypatch.setenv("BINANCE_ENV", "testnet")
    monkeypatch.setenv("BINANCE_MARKET_TYPE", "spot")
    settings = Settings.from_env()
    ex = ExchangeFactory.create_trading_client(settings)
    try:
        assert "testnet" in json.dumps(ex.urls["api"]).lower()
    finally:
        asyncio.run(ex.close())


# ── Descarga histórica (exchange simulado) ──────────────────────────────────
class FakeExchange:
    def __init__(self, candles):
        self.candles = candles
        self.calls = 0

    def parse_timeframe(self, tf):
        return 3600

    async def fetch_ohlcv(self, symbol, tf, since=None, limit=1000):
        self.calls += 1
        return [c for c in self.candles if c[0] >= since][:limit]


def test_fetch_range_pagina_sin_duplicados():
    step = 3_600_000
    candles = [[i * step, 1, 2, 0.5, 1.5, 10] for i in range(10)]
    ex = FakeExchange(candles)
    dl = HistoricalDataDownloader(ex, settings=None)
    df = asyncio.run(dl.fetch_range("BTC/USDT", "1h", since_ms=0, limit=3))
    assert len(df) == 10 and df.index.is_unique and ex.calls >= 4


def test_sync_descarta_ultima_vela_abierta(tmp_path):
    step = 3_600_000
    base = int(time.time() * 1000) // step * step      # inicio de la hora actual (vela abierta)
    candles = [[base - k * step, 1, 2, 0.5, 1.5, 10] for k in range(4, -1, -1)]
    dl = HistoricalDataDownloader(FakeExchange(candles), SimpleNamespace(raw_dir=tmp_path))
    df = asyncio.run(dl.sync("BTC/USDT", "1h", days=1))
    assert len(df) == 4 and (tmp_path / "BTCUSDT_1h.parquet").exists()
