from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from src.core.events import EventBus, ExecutionReportEvent
from src.core.logger import setup_logging
from src.data.exchange_factory import ExchangeFactory
from src.execution.ledger import ExecutionLedger, OrderStatusLedger
from src.execution.portfolio import ExchangePortfolioReconciler
from src.execution.user_stream import BinanceUserDataStream

log = logging.getLogger(__name__)


def _persist_portfolio(path: Path, states: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        symbol: {
            "symbol": state.symbol,
            "equity": state.equity,
            "weight": state.weight,
            "price": state.price,
            "timestamp": state.timestamp.isoformat(),
            "entry_price": state.entry_price,
            "other_exposure": state.other_exposure,
            "source": state.source.value,
        }
        for symbol, state in states.items()
    }, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


async def main() -> None:
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    if not settings.api_key or not settings.api_secret:
        raise RuntimeError("Configura las credenciales API de solo lectura para el entorno seleccionado.")

    exchange = ExchangeFactory.create_trading_client(settings)
    bus = EventBus()
    stream: BinanceUserDataStream | None = None
    try:
        await exchange.load_markets()
        reconciler = ExchangePortfolioReconciler(exchange, settings)
        execution_dir = settings.raw_dir.parent / "execution"
        ledger = ExecutionLedger(execution_dir / f"fills_{settings.market_type.value}.jsonl")
        order_ledger = OrderStatusLedger(
            execution_dir / f"orders_{settings.market_type.value}.jsonl"
        )
        stream = BinanceUserDataStream(settings, exchange, bus, ledger, order_ledger)
        snapshot_path = execution_dir / f"portfolio_{settings.market_type.value}.json"

        def report_fill(event: ExecutionReportEvent) -> None:
            if event.is_fill:
                log.info(
                    "[fill exchange] %s %s qty=%s price=%s fee=%s %s status=%s",
                    event.symbol, event.side, event.last_filled_quantity,
                    event.last_filled_price, event.commission,
                    event.commission_asset or "", event.order_status,
                )
            else:
                log.info(
                    "[orden exchange] %s id=%s tipo=%s estado=%s acumulado=%s",
                    event.symbol, event.order_id, event.execution_type,
                    event.order_status, event.cumulative_filled_quantity,
                )

        bus.subscribe(ExecutionReportEvent, report_fill)

        async def reconcile_periodically() -> None:
            while True:
                try:
                    states = await reconciler.fetch_states()
                    _persist_portfolio(snapshot_path, states)
                    for state in states.values():
                        log.info(
                            "[cartera exchange] %s equity=%s %s exposure=%+.4f "
                            "other_exposure=%.4f entry=%s",
                            state.symbol, state.equity, state.symbol.split(":")[0].split("/")[-1],
                            state.weight, state.other_exposure, state.entry_price,
                        )
                except Exception:
                    log.exception("Falló la conciliación de cartera; no se reutiliza un snapshot interno.")
                await asyncio.sleep(60)

        log.warning(
            "Solo lectura: el stream registra fills reales; la conciliación REST es la fuente "
            "de PortfolioState. Este proceso no envía ni cancela órdenes."
        )
        await asyncio.gather(stream.run(), reconcile_periodically())
    finally:
        if stream is not None:
            await stream.stop()
        await exchange.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nUser Data Stream detenido.")
