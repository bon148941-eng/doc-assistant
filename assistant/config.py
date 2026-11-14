"""Настройки приложения: читаются из переменных окружения и файла .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_DOCS_DIR = "sample_docs"
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.10


class ConfigError(Exception):
    """Некорректное значение в настройках."""


@dataclass(frozen=True)
class Config:
    api_key: str | None
    model: str
    docs_dir: Path
    top_k: int
    min_score: float


def _read_number(name: str, default, cast):
    """Читает числовую переменную окружения; пустое значение = значение по умолчанию."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise ConfigError(f"Переменная {name} должна быть числом, получено: {raw!r}") from None


def load_config() -> Config:
    """Собирает настройки. Относительный DOCS_DIR считается от корня проекта."""
    load_dotenv(PROJECT_ROOT / ".env")

    docs_dir = Path(os.getenv("DOCS_DIR", "").strip() or DEFAULT_DOCS_DIR)
    if not docs_dir.is_absolute():
        docs_dir = PROJECT_ROOT / docs_dir

    top_k = _read_number("TOP_K", DEFAULT_TOP_K, int)
    if top_k < 1:
        raise ConfigError("Переменная TOP_K должна быть не меньше 1")

    min_score = _read_number("MIN_SCORE", DEFAULT_MIN_SCORE, float)
    if not 0 <= min_score <= 1:
        raise ConfigError(f"Переменная MIN_SCORE должна быть от 0 до 1, получено: {min_score}")

    return Config(
        api_key=os.getenv("ANTHROPIC_API_KEY", "").strip() or None,
        model=os.getenv("CLAUDE_MODEL", "").strip() or DEFAULT_MODEL,
        docs_dir=docs_dir,
        top_k=top_k,
        min_score=min_score,
    )
