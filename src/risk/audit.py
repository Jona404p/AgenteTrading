from __future__ import annotations

import json
from pathlib import Path

from src.risk.models import RiskDecision


class RiskAuditLog:
    """JSONL append-only con todas las decisiones de riesgo que NO fueron un ALLOW trivial (auditoría)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, decision: RiskDecision) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(decision.to_dict(), ensure_ascii=False) + "\n")
