from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import websockets
from websockets.exceptions import ConnectionClosed

from config.settings import BinanceEnv, MarketType, Settings
from src.core.events import EventBus, ExecutionReportEvent
from src.data.streams import to_stream_symbol
from src.execution.ledger import ExecutionLedger, OrderStatusLedger

log = logging.getLogger(__name__)
_KEEPALIVE_SECONDS = 25 * 60
_SPOT_WS_API_URLS = {
    BinanceEnv.LIVE: "wss://ws-api.binance.com:443/ws-api/v3",
    BinanceEnv.TESTNET: "wss://ws-api.testnet.binance.vision/ws-api/v3",
}


def _as_float(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(data.get(key, default) or 0.0)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Campo inválido en reporte de ejecución: {key}") from error
    if not value == value or value in (float("inf"), float("-inf")):
        raise ValueError(f"Campo no finito en reporte de ejecución: {key}")
    return value


def _as_int(data: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(data.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Campo inválido en reporte de ejecución: {key}") from error


def parse_execution_report(
    payload: dict[str, Any],
    market_type: MarketType,
    resolve_symbol: Callable[[str], str] | None = None,
) -> ExecutionReportEvent | None:
    """Parse Binance Spot executionReport or USDⓈ-M Futures ORDER_TRADE_UPDATE."""
    event_type = payload.get("e")
    if market_type is MarketType.SPOT and event_type == "executionReport":
        data = payload
    elif market_type is MarketType.FUTURE and event_type == "ORDER_TRADE_UPDATE":
        data = payload.get("o")
        if not isinstance(data, dict):
            raise ValueError("ORDER_TRADE_UPDATE no contiene el objeto `o`.")
    else:
        return None

    market_id = data.get("s")
    if not isinstance(market_id, str) or not market_id:
        raise ValueError("El reporte de ejecución no contiene un símbolo.")
    symbol = resolve_symbol(market_id) if resolve_symbol else market_id
    commission_asset = data.get("N")
    return ExecutionReportEvent(
        symbol=symbol,
        market_id=market_id,
        order_id=_as_int(data, "i"),
        client_order_id=str(data.get("c", "")),
        side=str(data.get("S", "")),
        execution_type=str(data.get("x", "")),
        order_status=str(data.get("X", "")),
        last_filled_quantity=_as_float(data, "l"),
        last_filled_price=_as_float(data, "L"),
        cumulative_filled_quantity=_as_float(data, "z"),
        cumulative_quote_quantity=_as_float(data, "Z"),
        commission=_as_float(data, "n"),
        commission_asset=str(commission_asset) if commission_asset else None,
        trade_id=_as_int(data, "t", -1),
        transaction_time=_as_int(data, "T", _as_int(payload, "E", 0)),
        realized_pnl=_as_float(data, "rp"),
    )


class BinanceUserDataStream:
    """Read-only authenticated Binance user stream with reconnect and listen-key renewal."""

    def __init__(
        self,
        settings: Settings,
        exchange: Any,
        bus: EventBus,
        ledger: ExecutionLedger,
        order_ledger: OrderStatusLedger | None = None,
    ) -> None:
        if not settings.api_key or not settings.api_secret:
            raise ValueError("El User Data Stream requiere API key y secret.")
        self.settings = settings
        self.exchange = exchange
        self.bus = bus
        self.ledger = ledger
        self.order_ledger = order_ledger
        self._stop = asyncio.Event()
        self._ws: Any = None
        self._spot_subscribed = False
        self._symbol_map = {to_stream_symbol(s): s for s in settings.symbols}
        markets = getattr(exchange, "markets", None) or {}
        self._symbol_map.update({
            market_id.upper(): symbol for market_id, symbol in (
                (market.get("id"), market.get("symbol"))
                for market in markets.values()
            )
            if market_id and symbol
        })

    def _listen_key_method(self, operation: str) -> Callable[..., Any]:
        method_name = f"fapiPrivate{operation.title()}ListenKey"
        method = getattr(self.exchange, method_name, None)
        if method is None:
            raise RuntimeError(f"CCXT no expone el endpoint Binance {operation} listen key.")
        return method

    def _spot_subscription_request(self, request_id: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "apiKey": self.settings.api_key,
            "timestamp": int(time.time() * 1000),
        }
        payload = "&".join(f"{key}={params[key]}" for key in sorted(params))
        params["signature"] = hmac.new(
            self.settings.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "id": request_id,
            "method": "userDataStream.subscribe.signature",
            "params": params,
        }

    async def _create_listen_key(self) -> str:
        response = await self._listen_key_method("post")()
        listen_key = response.get("listenKey") if isinstance(response, dict) else None
        if not isinstance(listen_key, str) or not listen_key:
            raise RuntimeError("Binance no devolvió un listen key válido.")
        return listen_key

    async def _subscribe_spot(self, ws: Any) -> None:
        request_id = str(uuid4())
        await ws.send(json.dumps(self._spot_subscription_request(request_id)))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            try:
                response = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as error:
                raise RuntimeError("Binance devolvió una respuesta inválida al suscribir Spot.") from error
            if not isinstance(response, dict):
                raise RuntimeError("Binance devolvió una respuesta no válida al suscribir Spot.")
            if response.get("id") != request_id:
                await self._dispatch(raw)
                continue
            if response.get("status") != 200:
                error = response.get("error")
                code = error.get("code") if isinstance(error, dict) else None
                raise RuntimeError(
                    f"Binance rechazó la suscripción Spot (status={response.get('status')}, "
                    f"code={code}). Revisa la API key de Testnet y su permiso de lectura "
                    "para User Data Stream."
                )
            result = response.get("result")
            if not isinstance(result, dict) or not isinstance(result.get("subscriptionId"), int):
                raise RuntimeError("Binance no confirmó la suscripción Spot.")
            self._spot_subscribed = True
            return

    async def _run_spot_connection(self) -> None:
        url = _SPOT_WS_API_URLS[self.settings.env]
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            self._ws = ws
            await self._subscribe_spot(ws)
            log.info(
                "User Data Stream autenticado conectado (%s, Spot WebSocket API)",
                self.settings.env.value,
            )
            while not self._stop.is_set():
                await self._dispatch(await ws.recv())

    async def _keepalive(self, listen_key: str) -> None:
        await self._listen_key_method("put")({"listenKey": listen_key})

    async def _delete_listen_key(self, listen_key: str) -> None:
        try:
            await self._listen_key_method("delete")({"listenKey": listen_key})
        except Exception as error:
            message = str(error).replace(listen_key, "[REDACTED]")
            log.error("No se pudo cerrar el listen key (%s): %s", type(error).__name__, message)

    def _resolve_symbol(self, market_id: str) -> str:
        symbol = self._symbol_map.get(market_id.upper())
        if symbol:
            return symbol
        try:
            return self.exchange.safe_symbol(market_id)
        except (AttributeError, KeyError, TypeError, ValueError):
            log.warning("Símbolo de ejecución no reconocido por CCXT; se conserva el id")
            return market_id

    async def run(self) -> None:
        """Reconnect indefinitely using Spot WebSocket API or the Futures listen key."""
        backoff = 1
        while not self._stop.is_set():
            listen_key: str | None = None
            try:
                if self.settings.market_type is MarketType.SPOT:
                    self._spot_subscribed = False
                    await self._run_spot_connection()
                    backoff = 1
                else:
                    listen_key = await self._create_listen_key()
                    url = f"{self.settings.ws_base_url}/ws/{quote(listen_key, safe='')}"
                    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                        self._ws, backoff = ws, 1
                        log.info(
                            "User Data Stream autenticado conectado (%s)",
                            self.settings.market_type.value,
                        )
                        deadline = time.monotonic() + _KEEPALIVE_SECONDS
                        while not self._stop.is_set():
                            remaining = max(0.1, deadline - time.monotonic())
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                            except asyncio.TimeoutError:
                                await self._keepalive(listen_key)
                                deadline = time.monotonic() + _KEEPALIVE_SECONDS
                                continue
                            await self._dispatch(raw)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as error:
                if not self._stop.is_set():
                    if self._spot_subscribed:
                        backoff = 1
                    message = str(error)
                    if listen_key is not None:
                        message = message.replace(listen_key, "[REDACTED]")
                    message = message.replace(self.settings.api_key, "[REDACTED]")
                    message = message.replace(self.settings.api_secret, "[REDACTED]")
                    log.warning(
                        "User Data Stream interrumpido (%s: %s); reconexión en %ss",
                        type(error).__name__, message, backoff,
                    )
            except Exception as error:
                if self._stop.is_set():
                    break
                if self._spot_subscribed:
                    backoff = 1
                message = str(error)
                if listen_key is not None:
                    message = message.replace(listen_key, "[REDACTED]")
                message = message.replace(self.settings.api_key, "[REDACTED]")
                message = message.replace(self.settings.api_secret, "[REDACTED]")
                log.error(
                    "Error en User Data Stream autenticado (%s): %s",
                    type(error).__name__, message,
                )
            finally:
                self._ws = None
                self._spot_subscribed = False
                if listen_key is not None:
                    await self._delete_listen_key(listen_key)
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            await self._ws.close()

    async def _dispatch(self, raw: str | bytes) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Mensaje JSON inválido en User Data Stream")
            return
        if not isinstance(payload, dict):
            log.warning("Mensaje no-objeto en User Data Stream")
            return
        wrapped_event = payload.get("event")
        if isinstance(wrapped_event, dict):
            payload = wrapped_event
        if payload.get("e") == "eventStreamTerminated":
            log.warning("Binance terminó el User Data Stream; se reconectará.")
            if self._ws is not None:
                await self._ws.close()
            return
        if payload.get("e") == "listenKeyExpired":
            log.warning("Binance expiró el listen key; se renovará el stream")
            if self._ws is not None:
                await self._ws.close()
            return
        try:
            event = parse_execution_report(payload, self.settings.market_type, self._resolve_symbol)
        except (KeyError, TypeError, ValueError):
            log.exception("Reporte de ejecución inválido (%s)", payload.get("e"))
            return
        if event is None:
            log.debug("Evento de cuenta no ejecutable recibido: %s", payload.get("e"))
            return
        if self.order_ledger is not None:
            self.order_ledger.record(event)
        if event.is_fill:
            self.ledger.record(event)
        await self.bus.publish(event)
