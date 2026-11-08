import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.events import DriftEvent, EventBus


def test_drift_event_publishes_on_event_bus():
    bus = EventBus()
    received = []
    event = DriftEvent(
        bar="2026-10-11T12:00:00+00:00",
        symbol="BTC/USDT",
        level="WARN",
        n_drifted=2,
        n_features=14,
        drifted=("rsi", "macd"),
        reasons=("PSI exceeded threshold",),
        tstat=2.5,
        drawdown=0.03,
    )
    bus.subscribe(DriftEvent, received.append)

    asyncio.run(bus.publish(event))

    assert received == [event]
