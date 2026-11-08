import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.live_settings import DriftSettings
from src.live.drift import ModelDriftMonitor, gate_exposure_on_critical


def test_drift_status_reports_warmup_then_critical_for_shifted_features():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(500, 2))
    settings = DriftSettings(
        enabled=True,
        psi_threshold=0.1,
        critical_psi_threshold=0.5,
        critical_feature_fraction=0.5,
        max_drawdown=0.2,
    )
    monitor = ModelDriftMonitor(settings, ["rsi", "atr_pct"], reference, window=20)

    warmup = monitor.evaluate(reference[0], 0.0)
    assert warmup.level == "WARMUP"
    assert not warmup.entries_paused

    for row in np.full((20, 2), 10.0):
        critical = monitor.evaluate(row, 0.0)

    assert critical.level == "CRITICAL"
    assert critical.entries_paused
    assert critical.n_drifted == 2
    assert critical.psi_max is not None and critical.psi_max >= settings.critical_psi_threshold


def test_critical_drawdown_pauses_entries_even_when_features_match():
    reference = np.linspace(-2, 2, 100).reshape(-1, 1)
    monitor = ModelDriftMonitor(
        DriftSettings(enabled=True, max_drawdown=0.1),
        ["rsi"],
        reference,
        window=20,
    )

    status = None
    for row in reference[:20]:
        status = monitor.evaluate(row, 0.15)

    assert status is not None and status.level == "CRITICAL"
    assert status.entries_paused
    assert any("drawdown" in reason for reason in status.reasons)


@pytest.mark.parametrize(
    ("current", "requested", "expected"),
    [
        (0.0, 0.8, 0.0),
        (0.4, 0.8, 0.4),
        (0.4, 0.2, 0.2),
        (0.4, -0.8, 0.0),
        (-0.4, -0.8, -0.4),
        (-0.4, 0.2, 0.0),
    ],
)
def test_critical_entry_gate_only_allows_flatten_or_lower_same_side(
    current, requested, expected
):
    assert gate_exposure_on_critical(current, requested) == pytest.approx(expected)


def test_critical_entry_gate_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finita"):
        gate_exposure_on_critical(0.2, float("nan"))
