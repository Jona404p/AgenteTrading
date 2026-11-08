import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from src.core.logger import setup_logging
from src.data.exchange_factory import ExchangeFactory
from src.data.features import FeatureEngineer
from src.data.historical import HistoricalDataDownloader, symbol_to_filename


async def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga y calcula features de mercado.")
    parser.add_argument("--history-days", type=int, default=None,
                        help="días históricos objetivo (por defecto HISTORY_DAYS)")
    parser.add_argument("--refresh", action="store_true",
                        help="backfill del rango completo incluso si ya hay datos guardados")
    args = parser.parse_args()
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    history_days = args.history_days if args.history_days is not None else settings.history_days
    if history_days < 1:
        parser.error("--history-days debe ser mayor que cero")

    exchange = (
        ExchangeFactory.create_public_data_client(settings)
        if settings.historical_from_live
        else ExchangeFactory.create_trading_client(settings)
    )
    engineer = FeatureEngineer()
    try:
        downloader = HistoricalDataDownloader(exchange, settings)
        for symbol in settings.symbols:
            raw = await downloader.sync(symbol, settings.timeframe, history_days, refresh=args.refresh)
            feats = engineer.transform(raw)
            out = settings.processed_dir / f"{symbol_to_filename(symbol)}_{settings.timeframe}_features.parquet"
            feats.to_parquet(out)
            print(f"{symbol}: raw={len(raw)} filas | features={len(feats)} filas -> {out.name}")
    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
