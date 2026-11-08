from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


@dataclass(frozen=True, slots=True)
class Headline:
    source: str
    title: str
    published: datetime          # timezone-aware, UTC
    url: str = ""

    @property
    def uid(self) -> str:
        return hashlib.sha1(normalize_title(self.title).encode("utf-8")).hexdigest()[:16]


# ── Esquema que debe devolver el LLM (los textos de Field guían al modelo) ──
class HeadlineAssessment(BaseModel):
    index: int = Field(description="1-based number of the headline exactly as listed")
    score: float = Field(
        description="Expected short-term price impact: 0 = extremely bearish, "
                    "50 = neutral / no impact, 100 = extremely bullish"
    )
    relevance: float = Field(
        description="0 to 1: how directly this headline affects the asset or the "
                    "macro/risk environment that moves it (0 = unrelated)"
    )
    reason: str = Field(default="", description="At most 12 words")


class SentimentAnalysis(BaseModel):
    assessments: list[HeadlineAssessment]


@dataclass(frozen=True, slots=True)
class SentimentResult:
    asset: str
    score: float
    confidence: float
    n_headlines: int             # titulares relevantes en la ventana
    n_scored: int                # titulares con puntuación válida
    fear_greed: int | None
    drivers: tuple[str, ...]     # titulares que más movieron el score (auditoría)
    timestamp: datetime
    degraded: bool

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 3),
            "n_headlines": self.n_headlines,
            "n_scored": self.n_scored,
            "fear_greed": self.fear_greed,
            "drivers": list(self.drivers),
            "degraded": self.degraded,
        }
