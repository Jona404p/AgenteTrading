import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.logger import setup_logging


def test_setup_logging_keeps_console_and_adds_rotating_file(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))
    path = tmp_path / "logs" / "bot.log"

    setup_logging("DEBUG", path)

    assert path.parent.is_dir()
    assert captured["level"] == logging.DEBUG
    assert len(captured["handlers"]) == 2
    assert isinstance(captured["handlers"][0], logging.StreamHandler)
    file_handler = captured["handlers"][1]
    assert isinstance(file_handler, RotatingFileHandler)
    assert file_handler.baseFilename == str(path)
    assert file_handler.maxBytes == 5_000_000
    assert file_handler.backupCount == 3
    assert file_handler.encoding == "utf-8"
    file_handler.close()


def test_setup_logging_without_file_keeps_console_handler(monkeypatch):
    captured = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))

    setup_logging()

    assert len(captured["handlers"]) == 1
    assert isinstance(captured["handlers"][0], logging.StreamHandler)
