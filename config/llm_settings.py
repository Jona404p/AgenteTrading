from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from config.settings import ROOT_DIR  # importar esto ejecuta load_dotenv()

_DEFAULT_FEEDS = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/,"
    "https://cointelegraph.com/rss,"
    "https://www.federalreserve.gov/feeds/press_all.xml"
)


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float | None
    timeout: float
    max_retries: int
    ollama_base_url: str

    @classmethod
    def from_env(cls) -> "LLMSettings":
        temp_raw = os.getenv("LLM_TEMPERATURE", "0").strip()
        return cls(
            provider=os.getenv("LLM_PROVIDER", "anthropic").strip().lower(),
            model=os.getenv("LLM_MODEL", "").strip(),
            temperature=float(temp_raw) if temp_raw else None,
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )


@dataclass(frozen=True)
class SentimentSettings:
    lookback_hours: float
    max_headlines: int
    batch_size: int
    half_life_hours: float
    evidence_ref: float
    refresh_minutes: float
    cache_ttl_hours: float
    http_timeout: float
    llm_timeout: float
    rss_feeds: tuple[str, ...]
    store_dir: Path

    @classmethod
    def from_env(cls) -> "SentimentSettings":
        feeds = os.getenv("NEWS_RSS_FEEDS", _DEFAULT_FEEDS)
        return cls(
            lookback_hours=float(os.getenv("SENTIMENT_LOOKBACK_HOURS", "12")),
            max_headlines=int(os.getenv("SENTIMENT_MAX_HEADLINES", "30")),
            batch_size=int(os.getenv("SENTIMENT_BATCH_SIZE", "15")),
            half_life_hours=float(os.getenv("SENTIMENT_HALF_LIFE_HOURS", "6")),
            evidence_ref=float(os.getenv("SENTIMENT_EVIDENCE_REF", "3.0")),
            refresh_minutes=float(os.getenv("SENTIMENT_REFRESH_MINUTES", "15")),
            cache_ttl_hours=float(os.getenv("SENTIMENT_CACHE_TTL_HOURS", "24")),
            http_timeout=float(os.getenv("NEWS_HTTP_TIMEOUT", "10")),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            rss_feeds=tuple(u.strip() for u in feeds.split(",") if u.strip()),
            store_dir=ROOT_DIR / "data" / "sentiment",
        )
