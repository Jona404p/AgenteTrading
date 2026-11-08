import asyncio
import sys
from pathlib import Path

from ccxt.base.errors import AuthenticationError

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from src.core.logger import setup_logging
from src.data.exchange_factory import ExchangeFactory


async def verify_connection(settings: Settings, exchange, output=print) -> bool:
    if not settings.symbols:
        raise ValueError("No hay símbolos configurados en SYMBOLS.")

    await exchange.load_markets()
    symbol = settings.symbols[0]
    ticker = await exchange.fetch_ticker(symbol)
    output(f"[{settings.env.value}] {symbol} último precio: {ticker['last']}")

    if not settings.api_key and not settings.api_secret:
        output("Sin credenciales API: se omitió la verificación de balance.")
        return True
    if not settings.api_key or not settings.api_secret:
        output(
            "ERROR: falta la API key o el secret del entorno seleccionado; "
            "no se intentó consultar el balance."
        )
        return False

    try:
        balance = await exchange.fetch_balance()
    except AuthenticationError:
        output(
            "ERROR de autenticación al consultar balance (-2015 suele indicar API key "
            "inválida, entorno incorrecto, IP no autorizada o permisos insuficientes).\n"
            f"Comprueba que la key y el secret correspondan a BINANCE_ENV={settings.env.value}, "
            f"que tengan permiso de lectura de cuenta para {settings.market_type.value}, "
            "y que la IP actual esté permitida si activaste la lista blanca. "
            "Las credenciales de Testnet y producción son distintas. "
            "No compartas ni pegues tus claves."
        )
        return False

    free = {
        asset: amount
        for asset, amount in balance.get("free", {}).items()
        if amount
    }
    output(f"Balance libre: {free or 'vacío'}")
    return True


async def main() -> bool:
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    exchange = ExchangeFactory.create_trading_client(settings)
    try:
        return await verify_connection(settings, exchange)
    finally:
        await exchange.close()


if __name__ == "__main__":
    if not asyncio.run(main()):
        raise SystemExit(1)
