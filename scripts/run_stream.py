import asyncio
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from src.core.events import EventBus, KlineEvent, OrderBookEvent
from src.core.logger import setup_logging
from src.data.buffer import MarketDataBuffer
from src.data.features import FeatureEngineer
from src.data.historical import HistoricalDataDownloader
from src.data.streams import BinanceWebSocketClient


async def main() -> None:
    settings = Settings.from_env()
    setup_logging(settings.log_level)

    bus = EventBus()
    buffer = MarketDataBuffer(bus)
    engineer = FeatureEngineer()

    # Precarga del buffer con histórico guardado (si existe)
    loader = HistoricalDataDownloader(exchange=None, settings=settings)
    for symbol in settings.symbols:
        try:
            buffer.seed(symbol, loader.load(symbol, settings.timeframe))
        except FileNotFoundError:
            print(f"Sin histórico para {symbol}. Ejecuta scripts/download_data.py primero.")

    def on_closed_candle(ev: KlineEvent) -> None:
        if ev.is_closed:
            feats = engineer.transform(buffer.candles(ev.symbol), dropna=True)
            last = feats.iloc[-1] if not feats.empty else None
            print(f"🕯️  {ev.symbol} cierre={ev.close} | RSI={None if last is None else round(last.rsi, 2)}")

    last_print = {"t": 0.0}

    def on_book(ev: OrderBookEvent) -> None:
        if time.time() - last_print["t"] > 2:  # imprime cada 2 s para no saturar la consola
            last_print["t"] = time.time()
            print(f"📖 {ev.symbol} bid={ev.best_bid} ask={ev.best_ask} "
                  f"spread={ev.spread_bps:.2f}bps imbalance={ev.imbalance():+.2f}")

    bus.subscribe(KlineEvent, on_closed_candle)
    bus.subscribe(OrderBookEvent, on_book)

    ws = BinanceWebSocketClient(settings, bus)
    try:
        await ws.run()
    except asyncio.CancelledError:
        pass
    finally:
        await ws.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStream detenido.")
