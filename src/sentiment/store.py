from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.sentiment.models import SentimentResult


class SentimentStore:
    """Histórico append-only (JSONL) de cada score emitido. Base del monitor de drift (Fase 5)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def append(self, result: SentimentResult) -> None:
        path = self.directory / f"{result.asset}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def load(self, asset: str) -> pd.DataFrame:
        path = self.directory / f"{asset}.jsonl"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_json(path, lines=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")
