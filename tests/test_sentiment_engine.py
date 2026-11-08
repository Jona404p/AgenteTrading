import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.llm_settings import SentimentSettings
from src.sentiment.analyzer import SentimentEngine
from src.sentiment.models import Headline, HeadlineAssessment, SentimentAnalysis
from src.sentiment.store import SentimentStore


class FakeNews:
    def __init__(self, headlines):
        self.headlines = headlines

    async def fetch_headlines(self):
        return self.headlines

    async def fetch_fear_greed(self):
        return 40


class FakeLLM:
    """LLM simulado: puntúa 80 todo lo que contenga 'surge', 50 el resto. Cuenta las llamadas."""
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.fail:
            raise RuntimeError("LLM caído")
        body = messages[1][1]
        lines = [l for l in body.splitlines() if l[:1].isdigit()]
        return SentimentAnalysis(assessments=[
            HeadlineAssessment(index=int(l.split(".")[0]), score=80 if "surge" in l else 50, relevance=1.0)
            for l in lines
        ])


def _cfg(tmp_path):
    return SentimentSettings(
        lookback_hours=12, max_headlines=30, batch_size=15, half_life_hours=6, evidence_ref=2.0,
        refresh_minutes=15, cache_ttl_hours=24, http_timeout=5, llm_timeout=5,
        rss_feeds=(), store_dir=tmp_path,
    )


def _headlines():
    now = datetime.now(timezone.utc)
    return [
        Headline("t", "Bitcoin ETF inflows surge", now - timedelta(minutes=10)),
        Headline("t", "Bitcoin ETF flows surge again", now - timedelta(minutes=20)),
        Headline("t", "Bakery opens", now),
    ]


def test_flujo_completo_cache_y_store(tmp_path):
    llm = FakeLLM()
    engine = SentimentEngine(llm, FakeNews(_headlines()), _cfg(tmp_path), SentimentStore(tmp_path))
    r1 = asyncio.run(engine.analyze("BTC"))
    assert r1.n_headlines == 2 and r1.n_scored == 2 and not r1.degraded
    assert r1.score > 70 and r1.confidence > 0.9 and r1.fear_greed == 40

    asyncio.run(engine.analyze("BTC"))
    assert llm.calls == 1                      # segunda pasada: todo viene de la caché
    assert len(SentimentStore(tmp_path).load("BTC")) == 2


def test_fail_safe_si_el_llm_cae(tmp_path):
    engine = SentimentEngine(FakeLLM(fail=True), FakeNews(_headlines()), _cfg(tmp_path))
    r = asyncio.run(engine.analyze("BTC"))
    assert (r.score, r.confidence, r.degraded) == (50.0, 0.0, True)
