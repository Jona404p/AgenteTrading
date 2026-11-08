import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.sentiment.analyzer import aggregate
from src.sentiment.models import Headline, HeadlineAssessment
from src.sentiment.news_sources import is_relevant

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _h(title: str, hours_ago: float = 0) -> Headline:
    return Headline("test", title, NOW - timedelta(hours=hours_ago))


def _a(score: float, relevance: float = 1.0) -> HeadlineAssessment:
    return HeadlineAssessment(index=1, score=score, relevance=relevance)


def test_sin_evidencia_es_neutral():
    assert aggregate([], NOW, 6, 3.0)[:2] == (50.0, 0.0)


def test_contraccion_hacia_neutral_con_poca_evidencia():
    score, conf, _ = aggregate([(_h("x"), _a(90))], NOW, 6, 3.0)
    assert conf == pytest.approx(1 / 3)
    assert score == pytest.approx(50 + 40 / 3)


def test_titular_antiguo_pesa_menos():
    fresh, old = (_h("a"), _a(80)), (_h("b", hours_ago=12), _a(20))
    score, _, _ = aggregate([fresh, old], NOW, 6, 1.0)
    assert score > 50  # el reciente (alcista) domina sobre el viejo (bajista)


def test_relevancia_cero_no_aporta():
    score, conf, _ = aggregate([(_h("x"), _a(100, relevance=0.0))], NOW, 6, 3.0)
    assert (score, conf) == (50.0, 0.0)


def test_filtro_de_relevancia():
    assert is_relevant(_h("Bitcoin ETF inflows surge"), "BTC")
    assert is_relevant(_h("Fed signals rate cut"), "ETH")
    assert not is_relevant(_h("Local bakery opens new store"), "BTC")
    assert not is_relevant(_h("A new method for baking"), "ETH")  # 'eth' dentro de 'method'
