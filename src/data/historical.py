from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import Settings

log = logging.getLogger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def symbol_to_filename(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "_")


class HistoricalDataDownloader:
    def __init__(self, exchange, settings: Settings) -> None:
        self.exchange = exchange
        self.settings = settings

    def _path(self, symbol: str, timeframe: str) -> Path:
        return self.settings.raw_dir / f"{symbol_to_filename(symbol)}_{timeframe}.parquet"

    async def fetch_range(
        self, symbol: str, timeframe: str, since_ms: int, limit: int = 1000
    ) -> pd.DataFrame:
        """Descarga paginada desde `since_ms` hasta el presente."""
        tf_ms = self.exchange.parse_timeframe(timeframe) * 1000
        rows: list[list[float]] = []
        cursor = since_ms

        while True:
            batch = await self.exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
            if not batch:
                break
            rows.extend(batch)
            next_cursor = batch[-1][0] + tf_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < limit:
                break
            log.debug("%s %s: %d velas acumuladas", symbol, timeframe, len(rows))

        if not rows:
            return pd.DataFrame(columns=OHLCV_COLS)

        df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLS])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        return df

    async def sync(self, symbol: str, timeframe: str, days: int, refresh: bool = False) -> pd.DataFrame:
        """Update cached OHLCV; refresh=True also backfills the requested historical range."""
        path = self._path(symbol, timeframe)
        existing = pd.read_parquet(path) if path.exists() else None

        if existing is not None and not existing.empty and not refresh:
            since_ms = int(existing.index[-1].timestamp() * 1000)
            log.info("%s: reanudando desde %s", symbol, existing.index[-1])
        else:
            start = datetime.now(timezone.utc) - timedelta(days=days)
            since_ms = int(start.timestamp() * 1000)
            log.info("%s: %s de %d días", symbol,
                     "actualización histórica" if refresh and existing is not None else "descarga completa", days)

        new = await self.fetch_range(symbol, timeframe, since_ms)
        df = pd.concat([existing, new]) if existing is not None else new
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # La última vela puede estar abierta: se descarta para no entrenar con datos incompletos.
        tf_td = pd.Timedelta(milliseconds=self.exchange.parse_timeframe(timeframe) * 1000)
        if not df.empty and df.index[-1] + tf_td > pd.Timestamp.now(tz="UTC"):
            df = df.iloc[:-1]

        df.to_parquet(path)
        log.info("%s: %d velas guardadas en %s", symbol, len(df), path.name)
        return df

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(symbol, timeframe))
