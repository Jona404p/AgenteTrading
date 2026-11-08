import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ccxt.base.errors import AuthenticationError

from scripts.check_connection import verify_connection


def settings(*, api_key="key", api_secret="secret", env="testnet", market_type="spot"):
    return SimpleNamespace(
        symbols=("BTC/USDT",),
        api_key=api_key,
        api_secret=api_secret,
        env=SimpleNamespace(value=env),
        market_type=SimpleNamespace(value=market_type),
    )


class FakeExchange:
    def __init__(self, *, balance=None, balance_error=None):
        self.balance = balance or {"free": {"USDT": 10.0, "BTC": 0.0}}
        self.balance_error = balance_error
        self.loaded = False
        self.balance_calls = 0

    async def load_markets(self):
        self.loaded = True

    async def fetch_ticker(self, symbol):
        assert symbol == "BTC/USDT"
        return {"last": 100.0}

    async def fetch_balance(self):
        self.balance_calls += 1
        if self.balance_error:
            raise self.balance_error
        return self.balance


def test_authentication_error_is_reported_without_traceback_or_secrets():
    exchange = FakeExchange(balance_error=AuthenticationError(
        "binance {key=do-not-print, msg=Invalid API-key, IP, or permissions}"
    ))
    output = []

    result = asyncio.run(verify_connection(settings(), exchange, output.append))

    assert result is False
    assert exchange.loaded and exchange.balance_calls == 1
    assert any("último precio: 100.0" in line for line in output)
    message = "\n".join(output)
    assert "Testnet y producción son distintas" in message
    assert "do-not-print" not in message
    assert "Traceback" not in message


def test_absent_credentials_skip_private_balance_check():
    exchange = FakeExchange()
    output = []

    result = asyncio.run(verify_connection(
        settings(api_key="", api_secret=""), exchange, output.append
    ))

    assert result is True
    assert exchange.balance_calls == 0
    assert any("Sin credenciales API" in line for line in output)


def test_partial_credentials_do_not_attempt_balance():
    exchange = FakeExchange()
    output = []

    result = asyncio.run(verify_connection(
        settings(api_key="key", api_secret=""), exchange, output.append
    ))

    assert result is False
    assert exchange.balance_calls == 0
    assert any("falta la API key o el secret" in line for line in output)


def test_valid_credentials_show_only_nonzero_free_balances():
    exchange = FakeExchange()
    output = []

    result = asyncio.run(verify_connection(settings(), exchange, output.append))

    assert result is True
    assert any("Balance libre: {'USDT': 10.0}" in line for line in output)
