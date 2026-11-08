import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level: str = "INFO", file: Path | None = None) -> None:
    """Configure console logging and an optional rotating UTF-8 log file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if file is not None:
        file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(
            file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        ))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
