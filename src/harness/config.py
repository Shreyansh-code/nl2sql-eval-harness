"""Settings loaded from the environment. No secret is ever returned by repr-safe dumps."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

# Loaded once, with override=False so a real environment variable always wins over the
# file. That keeps CI and one-off runs from depending on someone's local .env.
load_dotenv(REPO_ROOT / ".env", override=False)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return default if raw is None else int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_opt_float(name: str) -> float | None:
    raw = _env(name)
    return None if raw is None else float(raw)


def _publishable_path(path: Path) -> str:
    """Repo-relative where possible, else just the final component.

    Committed run metadata must not carry `/Users/<name>/...`.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


@dataclass(frozen=True)
class Settings:
    generator_model: str | None
    judge_model: str | None
    generator_temperature: float | None
    generator_mode: str
    judge_mode: str
    openai_base_url: str | None
    openai_api_key: str | None
    data_dir: Path
    subset_path: Path
    max_concurrency: int
    sql_timeout_seconds: float
    max_rows: int
    label_sample_size: int
    langsmith_tracing: bool

    @property
    def databases_dir(self) -> Path:
        return self.data_dir / "dev_databases"

    def redacted(self) -> dict[str, object]:
        """Snapshot safe to commit into a run's meta.json.

        Absolute local paths are rewritten to repo-relative form: a public artifact
        should not carry the author's home directory.
        """
        return {
            "generator_model": self.generator_model,
            "judge_model": self.judge_model,
            "generator_temperature": self.generator_temperature,
            "generator_mode": self.generator_mode,
            "judge_mode": self.judge_mode,
            "openai_base_url": self.openai_base_url,
            "data_dir": _publishable_path(self.data_dir),
            "subset_path": _publishable_path(self.subset_path),
            "max_concurrency": self.max_concurrency,
            "sql_timeout_seconds": self.sql_timeout_seconds,
            "max_rows": self.max_rows,
            "label_sample_size": self.label_sample_size,
            "langsmith_tracing": self.langsmith_tracing,
        }


def load_settings() -> Settings:
    subset_path = Path(_env("SUBSET_PATH", str(REPO_ROOT / "subset.yaml")))
    if not subset_path.is_absolute():
        subset_path = REPO_ROOT / subset_path
    data_dir = Path(_env("DATA_DIR", str(REPO_ROOT / "data" / "bird")))
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    generator_mode = _env("GENERATOR_MODE", "standard") or "standard"
    judge_mode = _env("JUDGE_MODE", "batch") or "batch"
    if generator_mode not in {"standard", "batch"}:
        raise ValueError(f"GENERATOR_MODE must be standard|batch, got {generator_mode!r}")
    if judge_mode not in {"standard", "batch"}:
        raise ValueError(f"JUDGE_MODE must be standard|batch, got {judge_mode!r}")

    return Settings(
        generator_model=_env("GENERATOR_MODEL"),
        judge_model=_env("JUDGE_MODEL"),
        # Left unset by default. Reasoning models such as gpt-6-luna reject
        # temperature=0 and accept only their default, so sending one is a 400. Where a
        # model does accept it, set GENERATOR_TEMPERATURE=0 for reproducible sampling.
        generator_temperature=_env_opt_float("GENERATOR_TEMPERATURE"),
        generator_mode=generator_mode,
        judge_mode=judge_mode,
        openai_base_url=_env("OPENAI_BASE_URL"),
        openai_api_key=_env("OPENAI_API_KEY"),
        data_dir=data_dir,
        subset_path=subset_path,
        max_concurrency=_env_int("MAX_CONCURRENCY", 8),
        sql_timeout_seconds=float(_env_int("SQL_TIMEOUT_SECONDS", 5)),
        max_rows=_env_int("MAX_ROWS", 1000),
        label_sample_size=_env_int("LABEL_SAMPLE_SIZE", 50),
        langsmith_tracing=_env_bool("LANGSMITH_TRACING", False),
    )
