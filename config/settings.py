from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent


class BinanceEnv(str, Enum):
    TESTNET = "testnet"
    LIVE = "live"


class MarketType(str, Enum):
    SPOT = "spot"
    FUTURE = "future"


# Base de streams por (mercado, entorno)
_WS_BASE_URLS = {
    (MarketType.SPOT, BinanceEnv.LIVE): "wss://stream.binance.com:9443",
    (MarketType.SPOT, BinanceEnv.TESTNET): "wss://stream.testnet.binance.vision",
    (MarketType.FUTURE, BinanceEnv.LIVE): "wss://fstream.binance.com",
    (MarketType.FUTURE, BinanceEnv.TESTNET): "wss://stream.binancefuture.com",
}


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _int_or_default(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s debe ser un entero; se usará el valor seguro predeterminado %d.", name, default)
        return default


@dataclass(frozen=True)
class Settings:
    env: BinanceEnv
    market_type: MarketType
    api_key: str
    api_secret: str
    symbols: tuple[str, ...]
    timeframe: str
    history_days: int
    historical_from_live: bool
    ws_depth_levels: int
    log_level: str
    raw_dir: Path
    processed_dir: Path

    @property
    def ws_base_url(self) -> str:
        return _WS_BASE_URLS[(self.market_type, self.env)]

    @property
    def is_live(self) -> bool:
        return self.env is BinanceEnv.LIVE

    @classmethod
    def from_env(cls) -> "Settings":
        env = BinanceEnv(os.getenv("BINANCE_ENV", "testnet").strip().lower())
        market = MarketType(os.getenv("BINANCE_MARKET_TYPE", "spot").strip().lower())

        # Candado de seguridad para producción
        if env is BinanceEnv.LIVE and os.getenv("CONFIRM_LIVE_TRADING", "NO") != "YES":
            raise RuntimeError(
                "BINANCE_ENV=live requiere CONFIRM_LIVE_TRADING=YES en el .env. "
                "Operación abortada por seguridad."
            )

        prefix = "BINANCE_LIVE" if env is BinanceEnv.LIVE else "BINANCE_TESTNET"
        raw_symbols = os.getenv("SYMBOLS", "BTC/USDT")
        settings = cls(
            env=env,
            market_type=market,
            api_key=os.getenv(f"{prefix}_API_KEY", ""),
            api_secret=os.getenv(f"{prefix}_API_SECRET", ""),
            symbols=tuple(s.strip().upper() for s in raw_symbols.split(",") if s.strip()),
            timeframe=os.getenv("TIMEFRAME", "1h"),
            history_days=int(os.getenv("HISTORY_DAYS", "365")),
            historical_from_live=_bool("HISTORICAL_FROM_LIVE", "true"),
            ws_depth_levels=_int_or_default("WS_DEPTH_LEVELS", 20),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            raw_dir=ROOT_DIR / "data" / "raw",
            processed_dir=ROOT_DIR / "data" / "processed",
        )
        settings.raw_dir.mkdir(parents=True, exist_ok=True)
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        return settings
