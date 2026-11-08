from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from src.core.events import EventBus, SentimentEvent
from src.sentiment.analyzer import SentimentEngine
from src.sentiment.models import SentimentResult

log = logging.getLogger(__name__)


class SentimentService:
    """Refresca el sentimiento periódicamente y publica SentimentEvent en el bus."""

    def __init__(
        self, engine: SentimentEngine, bus: EventBus, assets: Sequence[str], interval_minutes: float
    ) -> None:
        self.engine, self.bus = engine, bus
        self.assets = list(assets)
        self.interval = interval_minutes * 60
        self._latest: dict[str, SentimentResult] = {}
        self._stop = asyncio.Event()

    def latest(self, asset: str) -> SentimentResult | None:
        return self._latest.get(asset)

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                results = await self.engine.analyze_many(self.assets)
                for asset, r in results.items():
                    self._latest[asset] = r
                    await self.bus.publish(SentimentEvent(
                        asset=asset, score=r.score, confidence=r.confidence,
                        n_scored=r.n_scored, fear_greed=r.fear_greed, degraded=r.degraded,
                        timestamp=int(r.timestamp.timestamp() * 1000),
                        drivers=r.drivers,
                    ))
            except Exception:
                log.exception("Error en el ciclo de sentimiento")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass
