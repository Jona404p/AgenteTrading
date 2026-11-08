from __future__ import annotations

import json
from pathlib import Path

from src.core.events import ExecutionReportEvent


def _event_record(event: ExecutionReportEvent) -> dict:
    return {
        "symbol": event.symbol,
        "market_id": event.market_id,
        "order_id": event.order_id,
        "client_order_id": event.client_order_id,
        "side": event.side,
        "execution_type": event.execution_type,
        "order_status": event.order_status,
        "last_filled_quantity": event.last_filled_quantity,
        "last_filled_price": event.last_filled_price,
        "cumulative_filled_quantity": event.cumulative_filled_quantity,
        "cumulative_quote_quantity": event.cumulative_quote_quantity,
        "commission": event.commission,
        "commission_asset": event.commission_asset,
        "trade_id": event.trade_id,
        "transaction_time": event.transaction_time,
        "realized_pnl": event.realized_pnl,
    }


class ExecutionLedger:
    """Append-only record of exchange fills, deduplicated by exchange trade ID."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[tuple[str, int, int]] = set()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as ledger:
                for line_number, line in enumerate(ledger, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        self._seen.add((
                            record["market_id"], int(record["order_id"]), int(record["trade_id"])
                        ))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"Registro de fills inválido en {self.path}:{line_number}"
                        ) from error

    def record(self, event: ExecutionReportEvent) -> bool:
        """Persist one actual fill; return False when Binance replays a known trade."""
        if not event.is_fill:
            return False
        if event.trade_id < 0:
            raise ValueError("Fill recibido sin trade_id válido; no se puede deduplicar.")
        key = (event.market_id, event.order_id, event.trade_id)
        if key in self._seen:
            return False
        with self.path.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(
                _event_record(event), ensure_ascii=False, allow_nan=False
            ) + "\n")
            ledger.flush()
        self._seen.add(key)
        return True


class OrderStatusLedger:
    """Append-only, replay-deduplicated record of exchange order lifecycle updates."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[tuple[str, int, str, str, float, int]] = set()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as ledger:
                for line_number, line in enumerate(ledger, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        self._seen.add(self._key(record))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"Registro de órdenes inválido en {self.path}:{line_number}"
                        ) from error

    @staticmethod
    def _key(record: dict) -> tuple[str, int, str, str, float, int]:
        return (
            record["market_id"],
            int(record["order_id"]),
            record["execution_type"],
            record["order_status"],
            float(record["cumulative_filled_quantity"]),
            int(record["trade_id"]),
        )

    def record(self, event: ExecutionReportEvent) -> bool:
        record = _event_record(event)
        key = self._key(record)
        if key in self._seen:
            return False
        with self.path.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            ledger.flush()
        self._seen.add(key)
        return True
