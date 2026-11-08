from __future__ import annotations

import logging
import os

from config.llm_settings import LLMSettings

log = logging.getLogger(__name__)


def _require_env(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"Falta {name} en el .env para el proveedor LLM seleccionado.")


class LLMFactory:
    """Devuelve un BaseChatModel de LangChain según LLM_PROVIDER. Imports perezosos:
    solo necesitas instalar el paquete del proveedor que uses."""

    @staticmethod
    def create(cfg: LLMSettings):
        if not cfg.model:
            raise RuntimeError("Define LLM_MODEL en el .env.")
        extra = {"temperature": cfg.temperature} if cfg.temperature is not None else {}
        log.info("LLM: provider=%s model=%s", cfg.provider, cfg.model)

        if cfg.provider == "anthropic":
            _require_env("ANTHROPIC_API_KEY")
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=cfg.model, timeout=cfg.timeout, max_retries=cfg.max_retries,
                max_tokens=2048, **extra,
            )
        if cfg.provider == "openai":
            _require_env("OPENAI_API_KEY")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=cfg.model, timeout=cfg.timeout, max_retries=cfg.max_retries, **extra,
            )
        if cfg.provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=cfg.model, base_url=cfg.ollama_base_url, **extra)

        raise ValueError(f"LLM_PROVIDER no soportado: {cfg.provider!r}")
