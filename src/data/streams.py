from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets
from websockets.exceptions import ConnectionClosed

from config.settings import Settings
from src.core.events import EventBus, KlineEvent, OrderBookEvent

log = logging.getLogger(__name__)


def to_stream_symbol(symbol: str) -> str:
    """'BTC/USDT' -> 'btcusdt'  |  'BTC/USDT:USDT' -> 'btcusdt'"""
    return symbol.split(":")[0].replace("/", "").lower()


class BinanceWebSocketClient:
    """Conecta a streams combinados de Binance y publica eventos en el EventBus."""

    def __init__(
        self,
        settings: Settings,
        bus: EventBus,
        symbols: list[str] | None = None,
        timeframe: str | None = None,
        klines: bool = True,
        depth: bool = True,
    ) -> None:
        self.settings = settings
        self.bus = bus
        self.symbols = symbols or list(settings.symbols)
        self.timeframe = timeframe or settings.timeframe
        self.klines, self.depth = klines, depth
        self._symbol_map = {to_stream_symbol(s): s for s in self.symbols}
        self._stop = asyncio.Event()
        self._ws = None

    def build_url(self) -> str:
        streams: list[str] = []
        for s in self._symbol_map:
            if self.klines:
                streams.append(f"{s}@kline_{self.timeframe}")
            if self.depth:
                streams.append(f"{s}@depth{self.settings.ws_depth_levels}@100ms")
        return f"{self.settings.ws_base_url}/stream?streams={'/'.join(streams)}"

    async def run(self) -> None:
        """Bucle con reconexión automática y backoff exponencial (Binance corta cada 24 h)."""
        url, backoff = self.build_url(), 1
        while not self._stop.is_set():
            try:
                log.info("Conectando WebSocket: %s", url)
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    self._ws, backoff = ws, 1
                    async for raw in ws:
                        await self._dispatch(raw)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                log.warning("WebSocket caído (%s). Reintento en %ss", exc, backoff)
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            await self._ws.close()

    async def _dispatch(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            stream, data = msg.get("stream", ""), msg.get("data", msg)
            stream_symbol = stream.split("@")[0]
            symbol = self._symbol_map.get(stream_symbol)
            if symbol is None:
                return

            if "@kline_" in stream:
                k = data["k"]
                await self.bus.publish(KlineEvent(
                    symbol=symbol, timeframe=k["i"], open_time=int(k["t"]),
                    open=float(k["o"]), high=float(k["h"]), low=float(k["l"]),
                    close=float(k["c"]), volume=float(k["v"]), is_closed=bool(k["x"]),
                ))
            elif "@depth" in stream:
                bids = data.get("bids") or data.get("b") or []   # spot / futures
                asks = data.get("asks") or data.get("a") or []
                if not bids or not asks:
                    return
                await self.bus.publish(OrderBookEvent(
                    symbol=symbol, timestamp=int(time.time() * 1000),
                    bids=tuple((float(p), float(q)) for p, q in bids),
                    asks=tuple((float(p), float(q)) for p, q in asks),
                ))
        except (KeyError, ValueError, TypeError):
            log.exception("Mensaje WebSocket inválido: %.200s", raw)
