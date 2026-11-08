from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.risk_settings import RiskSettings
from src.risk.models import BreakerState
from src.risk.guardrails import HARD_MAX_DAILY_LOSS_PCT

log = logging.getLogger(__name__)


class CircuitBreaker:
    """Interruptor de seguridad a nivel de cuenta. Estado persistido en JSON (sobrevive a reinicios).

    · DAILY_HALT: pérdida del día (vs. equity al inicio del día UTC) ≥ límite. Se libera al cambiar de día UTC.
    · DRAWDOWN_HALT: caída desde el máximo de equity ≥ límite. Se libera tras `drawdown_cooldown_hours`
      (si > 0) o manualmente con `reset()`; al liberar, el máximo se rebasa al equity actual.
    · STREAK_COOLDOWN: N pérdidas consecutivas → pausa de `loss_cooldown_minutes`.
    """

    def __init__(self, cfg: RiskSettings, state_path: Path | None = None) -> None:
        self.cfg = cfg
        self.path = state_path
        self.state = BreakerState.OK
        self.peak_equity: float | None = None
        self.day: str | None = None
        self.day_start_equity: float | None = None
        self.halted_at: datetime | None = None
        self.cooldown_until: datetime | None = None
        self._load()

    # ---------------------------------------------------------------- persistencia
    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text())
            self.state = BreakerState(d["state"])
            self.peak_equity, self.day, self.day_start_equity = d["peak_equity"], d["day"], d["day_start_equity"]
            self.halted_at = datetime.fromisoformat(d["halted_at"]) if d.get("halted_at") else None
            self.cooldown_until = datetime.fromisoformat(d["cooldown_until"]) if d.get("cooldown_until") else None
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            log.exception("Estado del circuit breaker ilegible; se bloquea por seguridad (%s)", self.path)
            self.state = BreakerState.DRAWDOWN_HALT
            self.peak_equity = None
            self.day = None
            self.day_start_equity = None
            self.halted_at = datetime.now(timezone.utc)
            self.cooldown_until = None
            self._save()

    def _save(self) -> None:
        if not self.path:
            return
        self.path.write_text(json.dumps({
            "state": self.state.value, "peak_equity": self.peak_equity, "day": self.day,
            "day_start_equity": self.day_start_equity,
            "halted_at": self.halted_at.isoformat() if self.halted_at else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        }))

    # ---------------------------------------------------------------- lógica
    def update(self, equity: float, now: datetime) -> BreakerState:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("El timestamp del circuit breaker debe incluir zona horaria.")
        now = now.astimezone(timezone.utc)
        if not math.isfinite(equity) or equity <= 0:
            self._trip(BreakerState.DRAWDOWN_HALT, now, "equity no válido o agotado")
            self._save()
            return self.state
        day = now.strftime("%Y-%m-%d")
        if self.day != day:                      # nuevo día UTC
            self.day, self.day_start_equity = day, equity
            if self.state is BreakerState.DAILY_HALT:
                log.info("Nuevo día UTC: se libera DAILY_HALT")
                self.state, self.halted_at = BreakerState.OK, None
        self.peak_equity = equity if self.peak_equity is None else max(self.peak_equity, equity)

        if self.state is BreakerState.DRAWDOWN_HALT and self.cfg.drawdown_cooldown_hours > 0 \
                and self.halted_at and now - self.halted_at >= timedelta(hours=self.cfg.drawdown_cooldown_hours):
            log.warning("Cooldown de drawdown cumplido: se libera y se rebasa el máximo de equity")
            self.reset(equity)
        if self.state is BreakerState.STREAK_COOLDOWN and self.cooldown_until and now >= self.cooldown_until:
            self.state, self.cooldown_until = BreakerState.OK, None

        if self.state in (BreakerState.OK, BreakerState.STREAK_COOLDOWN):
            dd = 1.0 - equity / self.peak_equity if self.peak_equity else 0.0
            daily = 1.0 - equity / self.day_start_equity if self.day_start_equity else 0.0
            if dd >= self.cfg.max_drawdown_pct:
                self._trip(BreakerState.DRAWDOWN_HALT, now, f"drawdown {dd:.1%}")
            configured_daily_limit = self.cfg.max_daily_loss_pct
            daily_limit = (
                min(configured_daily_limit, HARD_MAX_DAILY_LOSS_PCT)
                if math.isfinite(configured_daily_limit)
                else 0.0
            )
            if daily >= daily_limit and self.state is not BreakerState.DRAWDOWN_HALT:
                self._trip(BreakerState.DAILY_HALT, now, f"pérdida diaria {daily:.1%}")
        self._save()
        return self.state

    def register_trade(self, consecutive_losses: int, now: datetime) -> None:
        if (self.cfg.max_consecutive_losses > 0 and consecutive_losses >= self.cfg.max_consecutive_losses
                and self.state is BreakerState.OK):
            self.cooldown_until = now.astimezone(timezone.utc) + timedelta(minutes=self.cfg.loss_cooldown_minutes)
            self._trip(BreakerState.STREAK_COOLDOWN, now, f"{consecutive_losses} pérdidas seguidas")
            self._save()

    def reset(self, equity: float | None = None) -> None:
        """Reinicio manual (o por cooldown). Rebasa el máximo de equity para no volver a saltar al instante."""
        self.state, self.halted_at, self.cooldown_until = BreakerState.OK, None, None
        if equity is not None:
            self.peak_equity = equity
        self._save()

    def _trip(self, state: BreakerState, now: datetime, why: str) -> None:
        log.error("CIRCUIT BREAKER → %s (%s)", state.value, why)
        self.state, self.halted_at = state, now.astimezone(timezone.utc)
