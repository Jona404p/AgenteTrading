import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.llm_settings import LLMSettings, SentimentSettings
from config.settings import Settings
from src.core.events import EventBus, SentimentEvent
from src.core.logger import setup_logging
from src.sentiment.analyzer import SentimentEngine
from src.sentiment.llm_factory import LLMFactory
from src.sentiment.news_sources import NewsAggregator
from src.sentiment.service import SentimentService
from src.sentiment.store import SentimentStore


async def main(loop_mode: bool) -> None:
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    s_cfg = SentimentSettings.from_env()

    llm = LLMFactory.create(LLMSettings.from_env())
    engine = SentimentEngine(llm, NewsAggregator(s_cfg), s_cfg, SentimentStore(s_cfg.store_dir))
    assets = sorted({sym.split("/")[0] for sym in settings.symbols})

    if not loop_mode:
        for asset, r in (await engine.analyze_many(assets)).items():
            print(f"\n{asset}: score={r.score:.1f} confianza={r.confidence:.2f} "
                  f"({r.n_scored}/{r.n_headlines} titulares) F&G={r.fear_greed} "
                  f"{'⚠️ DEGRADADO' if r.degraded else ''}")
            for d in r.drivers:
                print("   ", d)
        return

    bus = EventBus()
    bus.subscribe(SentimentEvent, lambda e: print(
        f"📰 {e.asset} score={e.score:.1f} conf={e.confidence:.2f} F&G={e.fear_greed} degraded={e.degraded}"
    ))
    service = SentimentService(engine, bus, assets, s_cfg.refresh_minutes)
    try:
        await service.run()
    finally:
        service.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Refresco periódico (Ctrl+C para salir)")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.loop))
    except KeyboardInterrupt:
        print("\nDetenido.")
