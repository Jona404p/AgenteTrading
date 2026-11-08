from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from config.settings import MarketType, Settings
from src.risk.models import PortfolioSource, PortfolioState


class PortfolioReconciliationError(RuntimeError):
    """Raised when exchange snapshots are incomplete or cannot be valued safely."""


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PortfolioReconciliationError(f"El exchange devolvió {field} inválido.") from error
    if not math.isfinite(result):
        raise PortfolioReconciliationError(f"El exchange devolvió {field} no finito.")
    return result


class ExchangePortfolioReconciler:
    """Build risk snapshots only from authenticated exchange balance and position data."""

    def __init__(self, exchange: Any, settings: Settings) -> None:
        if not settings.api_key or not settings.api_secret:
            raise ValueError("La conciliación requiere credenciales del exchange.")
        if not settings.symbols:
            raise ValueError("La conciliación requiere al menos un símbolo configurado.")
        self.exchange = exchange
        self.settings = settings

    async def fetch_states(self) -> dict[str, PortfolioState]:
        if not getattr(self.exchange, "markets", None):
            await self.exchange.load_markets()
        if self.settings.market_type is MarketType.SPOT:
            return await self._fetch_spot_states()
        return await self._fetch_future_states()

    def _quote_asset(self) -> str:
        pairs = [symbol.split(":")[0].split("/") for symbol in self.settings.symbols]
        if any(len(pair) != 2 or not all(pair) for pair in pairs):
            raise PortfolioReconciliationError(
                "La conciliación exige símbolos con formato BASE/QUOTE."
            )
        quotes = {pair[1] for pair in pairs}
        if len(quotes) != 1:
            raise PortfolioReconciliationError(
                "La conciliación exige símbolos con formato BASE/QUOTE y una única divisa quote."
            )
        return next(iter(quotes))

    async def _ticker_price(self, symbol: str) -> float:
        ccxt_symbol = (
            self.exchange.market(symbol)["symbol"]
            if self.settings.market_type is MarketType.FUTURE
            else symbol
        )
        ticker = await self.exchange.fetch_ticker(ccxt_symbol)
        price = _number(ticker.get("last"), f"precio de {symbol}")
        if price <= 0:
            raise PortfolioReconciliationError(f"Precio no positivo para {symbol}.")
        return price

    async def _fetch_spot_states(self) -> dict[str, PortfolioState]:
        quote = self._quote_asset()
        balance = await self.exchange.fetch_balance()
        totals = balance.get("total")
        if not isinstance(totals, dict) or quote not in totals:
            raise PortfolioReconciliationError(
                f"El snapshot spot no incluye el saldo total de {quote}."
            )
        amounts = {
            asset: _number(amount, f"saldo de {asset}")
            for asset, amount in totals.items()
            if amount is not None
        }
        if any(amount < 0 for amount in amounts.values()):
            raise PortfolioReconciliationError("El snapshot spot contiene saldos negativos.")
        tickers: dict[str, float] = {}
        for asset, amount in amounts.items():
            if asset == quote or amount == 0:
                continue
            market_symbol = f"{asset}/{quote}"
            if market_symbol not in self.exchange.markets:
                raise PortfolioReconciliationError(
                    f"No se puede conciliar el saldo {asset}: falta el mercado "
                    f"{market_symbol}. Se descarta el snapshot completo para no calcular "
                    "equity parcial ni reutilizar un estado anterior."
                )
            tickers[market_symbol] = await self._ticker_price(market_symbol)
        equity = amounts[quote] + sum(
            amount * tickers[f"{asset}/{quote}"]
            for asset, amount in amounts.items()
            if asset != quote and amount != 0
        )
        if not math.isfinite(equity) or equity <= 0:
            raise PortfolioReconciliationError("El equity spot total no es positivo.")
        now = datetime.now(timezone.utc)
        states: dict[str, PortfolioState] = {}
        for symbol in self.settings.symbols:
            base = symbol.split(":")[0].split("/")[0]
            price = await self._ticker_price(symbol)
            quantity = amounts.get(base, 0.0)
            other_exposure = sum(
                abs(amount * tickers[f"{asset}/{quote}"])
                for asset, amount in amounts.items()
                if asset != quote and asset != base and amount != 0
            ) / equity
            states[symbol] = PortfolioState(
                symbol=symbol,
                equity=equity,
                weight=quantity * price / equity,
                price=price,
                timestamp=now,
                entry_price=None,
                other_exposure=other_exposure,
                source=PortfolioSource.EXCHANGE,
            )
        return states

    async def _fetch_future_states(self) -> dict[str, PortfolioState]:
        quote = self._quote_asset()
        balance = await self.exchange.fetch_balance()
        totals = balance.get("total")
        if not isinstance(totals, dict) or quote not in totals:
            raise PortfolioReconciliationError(
                f"El snapshot futures no incluye el equity total de {quote}."
            )
        equity = _number(totals[quote], f"equity de {quote}")
        if equity <= 0:
            raise PortfolioReconciliationError("El equity futures no es positivo.")
        ccxt_symbols = {
            symbol: self.exchange.market(symbol)["symbol"] for symbol in self.settings.symbols
        }
        positions = await self.exchange.fetch_positions()
        configured_by_pair = {
            symbol.split(":")[0]: symbol for symbol in self.settings.symbols
        }
        by_symbol: dict[str, dict[str, Any]] = {}
        for position in positions:
            exchange_symbol = position.get("symbol")
            if not isinstance(exchange_symbol, str) or not exchange_symbol:
                raise PortfolioReconciliationError(
                    "Una posición futures no incluye el símbolo del exchange."
                )
            contracts = _number(position.get("contracts", 0), f"contratos de {exchange_symbol}")
            if contracts == 0:
                continue
            side = position.get("side")
            if side not in {"long", "short"}:
                raise PortfolioReconciliationError(
                    f"No se pudo resolver el lado de la posición futures {exchange_symbol}."
                )
            if exchange_symbol in by_symbol:
                raise PortfolioReconciliationError(
                    f"Hay posiciones múltiples/hedge para {exchange_symbol}; "
                    "requiere conciliación hedge-mode."
                )
            market = self.exchange.market(exchange_symbol)
            market_quote = market.get("quote")
            if market_quote != quote:
                raise PortfolioReconciliationError(
                    f"La posición {exchange_symbol} no está denominada en {quote}."
                )
            contract_size = _number(
                market.get("contractSize", 1), f"contractSize de {exchange_symbol}"
            )
            if contract_size <= 0:
                raise PortfolioReconciliationError(
                    f"contractSize no positivo para {exchange_symbol}."
                )
            mark_price = (
                _number(position["markPrice"], f"mark price de {exchange_symbol}")
                if position.get("markPrice") is not None
                else await self._ticker_price(exchange_symbol)
            )
            if mark_price <= 0:
                raise PortfolioReconciliationError(
                    f"Mark price no positivo para {exchange_symbol}."
                )
            signed_contracts = contracts if side == "long" else -contracts
            reported_notional = position.get("notional")
            notional = (
                abs(_number(reported_notional, f"notional de {exchange_symbol}"))
                if reported_notional is not None
                else contracts * contract_size * mark_price
            )
            by_symbol[exchange_symbol] = {
                "configured_symbol": configured_by_pair.get(exchange_symbol.split(":")[0]),
                "contracts": signed_contracts,
                "notional": math.copysign(notional, signed_contracts),
                "entry_price": (
                    _number(position["entryPrice"], f"precio de entrada de {exchange_symbol}")
                    if position.get("entryPrice") is not None else None
                ),
                "mark_price": mark_price,
            }

        prices = {
            symbol: await self._ticker_price(ccxt_symbols[symbol])
            for symbol in self.settings.symbols
        }
        notionals = {symbol: pos["notional"] for symbol, pos in by_symbol.items()}
        now = datetime.now(timezone.utc)
        states: dict[str, PortfolioState] = {}
        for symbol in self.settings.symbols:
            exchange_symbol = ccxt_symbols[symbol]
            position = by_symbol.get(exchange_symbol)
            notional = position["notional"] if position else 0.0
            states[symbol] = PortfolioState(
                symbol=symbol,
                equity=equity,
                weight=notional / equity,
                price=prices[symbol],
                timestamp=now,
                entry_price=position["entry_price"] if position else None,
                other_exposure=sum(
                    abs(value) for other, value in notionals.items() if other != exchange_symbol
                ) / equity,
                source=PortfolioSource.EXCHANGE,
            )
        return states
