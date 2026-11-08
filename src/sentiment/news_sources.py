from __future__ import annotations

import asyncio
import calendar
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from urllib.parse import urlparse

import aiohttp
import feedparser

from config.llm_settings import SentimentSettings
from src.sentiment.models import Headline

log = logging.getLogger(__name__)

_CTRL = re.compile(r"[\x00-\x1f\x7f<>]")


def clean_text(text: str, max_len: int = 300) -> str:
    """Sanitiza texto externo: sin caracteres de control ni '<' '>' (evita cerrar nuestras etiquetas)."""
    return re.sub(r"\s+", " ", _CTRL.sub(" ", text)).strip()[:max_len]


# ── Filtro de relevancia por activo ─────────────────────────────────────────
ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin", "btc"),
    "ETH": ("ethereum", "ether", "eth"),
    "BNB": ("bnb", "binance coin"),
    "SOL": ("solana",),
    "XRP": ("xrp", "ripple"),
    "ADA": ("cardano",),
    "DOGE": ("dogecoin", "doge"),
}
_GENERAL_TERMS = (
    "crypto", "cryptocurrency", "stablecoin", "binance", "etf", "sec", "regulation",
    "fed", "federal reserve", "fomc", "inflation", "cpi", "interest rates?", "rate cut",
    "rate hike", "tariffs?", "recession", "treasury", "jobs report", "dollar",
)


def _compile(terms: tuple[str, ...]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(terms) + r")\b", re.IGNORECASE)


_GENERAL_RE = _compile(_GENERAL_TERMS)
_asset_re_cache: dict[str, re.Pattern] = {}


def is_relevant(headline: Headline, asset: str) -> bool:
    if asset not in _asset_re_cache:
        aliases = ASSET_ALIASES.get(asset.upper(), (asset.lower(),))
        _asset_re_cache[asset] = _compile(tuple(re.escape(a) for a in aliases))
    return bool(_asset_re_cache[asset].search(headline.title) or _GENERAL_RE.search(headline.title))


# ── Fuentes ─────────────────────────────────────────────────────────────────
class NewsSource(ABC):
    name: str

    @abstractmethod
    async def fetch(self, session: aiohttp.ClientSession) -> list[Headline]: ...


class RSSNewsSource(NewsSource):
    def __init__(self, url: str) -> None:
        self.url = url
        self.name = urlparse(url).netloc

    async def fetch(self, session: aiohttp.ClientSession) -> list[Headline]:
        async with session.get(self.url) as resp:
            resp.raise_for_status()
            text = await resp.text()
        parsed = await asyncio.to_thread(feedparser.parse, text)  # parseo CPU fuera del event loop

        out: list[Headline] = []
        for e in parsed.entries:
            title = clean_text(e.get("title", ""))
            if not title:
                continue
            ts = e.get("published_parsed") or e.get("updated_parsed")
            published = (
                datetime.fromtimestamp(calendar.timegm(ts), tz=timezone.utc)
                if ts else datetime.now(timezone.utc)
            )
            out.append(Headline(self.name, title, published, e.get("link", "")))
        return out


class NewsAggregator:
    """Consulta todas las fuentes en paralelo, tolera fallos parciales y deduplica."""

    FNG_URL = "https://api.alternative.me/fng/?limit=1"

    def __init__(self, cfg: SentimentSettings, sources: list[NewsSource] | None = None) -> None:
        self.cfg = cfg
        self.sources = sources or [RSSNewsSource(u) for u in cfg.rss_feeds]

    def _session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.cfg.http_timeout),
            headers={"User-Agent": "closed-loop-trading-agent/0.1"},
        )

    async def fetch_headlines(self) -> list[Headline]:
        async with self._session() as session:
            results = await asyncio.gather(
                *(s.fetch(session) for s in self.sources), return_exceptions=True
            )
        seen: dict[str, Headline] = {}
        for src, res in zip(self.sources, results):
            if isinstance(res, Exception):
                log.warning("Fuente %s falló: %s", src.name, res)
                continue
            for h in res:
                seen.setdefault(h.uid, h)
        return list(seen.values())

    async def fetch_fear_greed(self) -> int | None:
        try:
            async with self._session() as session, session.get(self.FNG_URL) as resp:
                resp.raise_for_status()
                payload = await resp.json()
            return int(payload["data"][0]["value"])
        except Exception as exc:
            log.warning("Fear & Greed no disponible: %s", exc)
            return None
