from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from config.settings import ROOT_DIR
from config.rl_settings import CostSettings


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class ExecutionSettings:
    min_rebalance: float = 0.05

    @classmethod
    def from_env(cls) -> "ExecutionSettings":
        value = _float("EXECUTION_MIN_REBALANCE", 0.05)
        if not 0 <= value <= 2:
            raise ValueError("EXECUTION_MIN_REBALANCE debe estar entre 0 y 2.")
        return cls(min_rebalance=value)


@dataclass(frozen=True)
class DriftSettings:
    enabled: bool = True
    psi_threshold: float = 0.25
    critical_psi_threshold: float = 0.5
    critical_feature_fraction: float = 0.5
    max_drawdown: float = 0.20

    @classmethod
    def from_env(cls) -> "DriftSettings":
        psi = _float("DRIFT_PSI_THRESHOLD", 0.25)
        critical_psi = _float("DRIFT_CRITICAL_PSI_THRESHOLD", 0.5)
        critical_fraction = _float("DRIFT_CRITICAL_FEATURE_FRACTION", 0.5)
        drawdown = _float("DRIFT_MAX_DRAWDOWN", 0.20)
        if psi <= 0 or critical_psi < psi or not 0 < critical_fraction <= 1 or not 0 < drawdown < 1:
            raise ValueError("Los umbrales de drift deben ser positivos y el drawdown menor que 1.")
        return cls(
            enabled=_bool("DRIFT_ENABLED", True),
            psi_threshold=psi,
            critical_psi_threshold=critical_psi,
            critical_feature_fraction=critical_fraction,
            max_drawdown=drawdown,
        )


@dataclass(frozen=True)
class LiveSettings:
    state_dir: Path
    models_root: Path
    warmup_candles: int
    paper_initial_cash: float
    shutdown_timeout_seconds: float
    flatten_on_exit: bool
    execution: ExecutionSettings
    drift: DriftSettings

    @classmethod
    def from_env(cls) -> "LiveSettings":
        state_dir = Path(os.getenv("LIVE_STATE_DIR", str(ROOT_DIR / "data" / "live")))
        models_root = Path(os.getenv("LIVE_MODELS_ROOT", str(ROOT_DIR / "data" / "models")))
        warmup = _int("LIVE_WARMUP_CANDLES", 400)
        initial_cash = _float("PAPER_INITIAL_CASH", 10_000.0)
        shutdown_timeout = _float("LIVE_SHUTDOWN_TIMEOUT_SECONDS", 30.0)
        if warmup < 1:
            raise ValueError("LIVE_WARMUP_CANDLES debe ser mayor que cero.")
        if initial_cash <= 0:
            raise ValueError("PAPER_INITIAL_CASH debe ser mayor que cero.")
        if shutdown_timeout <= 0:
            raise ValueError("LIVE_SHUTDOWN_TIMEOUT_SECONDS debe ser mayor que cero.")
        state_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            state_dir=state_dir,
            models_root=models_root,
            warmup_candles=warmup,
            paper_initial_cash=initial_cash,
            shutdown_timeout_seconds=shutdown_timeout,
            flatten_on_exit=_bool("LIVE_FLATTEN_ON_EXIT"),
            execution=ExecutionSettings.from_env(),
            drift=DriftSettings.from_env(),
        )
