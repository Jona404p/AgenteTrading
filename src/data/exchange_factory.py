from __future__ import annotations

import logging

import ccxt.async_support as ccxt

from config.settings import BinanceEnv, MarketType, Settings

log = logging.getLogger(__name__)


class ExchangeFactory:
    """Crea clientes ccxt de Binance según el entorno configurado en .env."""

    @staticmethod
    def _options(market_type: MarketType) -> dict:
        return {
            "enableRateLimit": True,
            "options": {
                "defaultType": market_type.value,   # "spot" | "future"
                "adjustForTimeDifference": True,
            },
        }

    @classmethod
    def create_public_data_client(cls, settings: Settings) -> ccxt.binance:
        """Cliente SIN credenciales contra producción. Solo para datos de mercado públicos."""
        log.info("Cliente público (producción, solo lectura) | %s", settings.market_type.value)
        return ccxt.binance(cls._options(settings.market_type))

    @classmethod
    def create_trading_client(cls, settings: Settings) -> ccxt.binance:
        """Cliente autenticado según BINANCE_ENV (testnet o live)."""
        config = cls._options(settings.market_type)
        config["apiKey"] = settings.api_key
        config["secret"] = settings.api_secret
        exchange = ccxt.binance(config)

        if settings.env is BinanceEnv.TESTNET:
            exchange.set_sandbox_mode(True)
            log.warning("Cliente de trading en BINANCE TESTNET (%s)", settings.market_type.value)
        else:
            log.warning("⚠️  Cliente de trading en BINANCE REAL (%s)", settings.market_type.value)
        return exchange
