from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

from src.core.events import EventBus, KlineEvent, OrderBookEvent


class MarketDataBuffer:
    """Observer que mantiene velas cerradas y el último order book por símbolo."""

    def __init__(self, bus: EventBus, max_len: int = 1500) -> None:
        self._candles: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_len))
        self._books: dict[str, OrderBookEvent] = {}
        bus.subscribe(KlineEvent, self._on_kline)
        bus.subscribe(OrderBookEvent, self._on_book)

    def seed(self, symbol: str, df: pd.DataFrame) -> None:
        """Precarga con histórico para que los indicadores tengan ventana desde el inicio."""
        q = self._candles[symbol]
        for ts, r in df.tail(q.maxlen).iterrows():
            self._upsert(symbol, ts, r)

    def upsert(self, symbol: str, rows: pd.Series | pd.DataFrame) -> None:
        """Insert or replace OHLCV rows, using their index values as candle timestamps."""
        if isinstance(rows, pd.DataFrame):
            for timestamp, row in rows.iterrows():
                self._upsert(symbol, timestamp, row)
            return
        if rows.name is None:
            raise ValueError("La fila OHLCV debe tener timestamp como índice.")
        self._upsert(symbol, rows.name, rows)

    def _upsert(self, symbol: str, timestamp, row: pd.Series) -> None:
        candle = (
            pd.Timestamp(timestamp),
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"],
        )
        q = self._candles[symbol]
        candles = list(q)
        for index, existing in enumerate(candles):
            if existing[0] == candle[0]:
                candles[index] = candle
                break
        else:
            candles.append(candle)
        q.clear()
        q.extend(sorted(candles, key=lambda item: item[0])[-q.maxlen:])

    def _on_kline(self, ev: KlineEvent) -> None:
        if not ev.is_closed:
            return
        ts = pd.Timestamp(ev.open_time, unit="ms", tz="UTC")
        self._upsert(ev.symbol, ts, pd.Series({
            "open": ev.open,
            "high": ev.high,
            "low": ev.low,
            "close": ev.close,
            "volume": ev.volume,
        }))

    def _on_book(self, ev: OrderBookEvent) -> None:
        self._books[ev.symbol] = ev

    def candles(self, symbol: str) -> pd.DataFrame:
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        return pd.DataFrame(list(self._candles[symbol]), columns=cols).set_index("timestamp")

    def order_book(self, symbol: str) -> OrderBookEvent | None:
        return self._books.get(symbol)
