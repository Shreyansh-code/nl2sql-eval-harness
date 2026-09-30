"""The generation port.

The rest of the pipeline depends on this interface and not on any provider, so the
standard and batch adapters are interchangeable and a run can be reproduced from
`generations.jsonl` without calling a model again.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from ..dataset.models import HarnessQuestion


class GenerationStatus(StrEnum):
    OK = "ok"
    UNPARSEABLE = "unparseable"
    EMPTY = "empty"
    API_ERROR = "api_error"


@dataclass(frozen=True)
class Generation:
    """One attempt at one question. `sql` is None whenever parsing failed."""

    question_id: str
    db_id: str
    status: GenerationStatus
    sql: str | None
    raw_response: str
    model: str
    prompt_version: str
    cache_key: str
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Whether this generation can be executed and scored."""
        return self.status is GenerationStatus.OK and bool(self.sql)

    def to_row(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "db_id": self.db_id,
            "status": str(self.status),
            "sql": self.sql,
            "raw_response": self.raw_response,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "cache_key": self.cache_key,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
            "extras": self.extras,
        }

    @classmethod
    def from_row(cls, row: dict) -> Generation:
        return cls(
            question_id=row["question_id"],
            db_id=row["db_id"],
            status=GenerationStatus(row["status"]),
            sql=row.get("sql"),
            raw_response=row.get("raw_response", ""),
            model=row.get("model", ""),
            prompt_version=row.get("prompt_version", ""),
            cache_key=row.get("cache_key", ""),
            latency_ms=int(row.get("latency_ms") or 0),
            input_tokens=int(row.get("input_tokens") or 0),
            output_tokens=int(row.get("output_tokens") or 0),
            error=row.get("error"),
            extras=row.get("extras") or {},
        )


class SqlGenerator(Protocol):
    model: str
    prompt_version: str

    def generate(self, questions: Sequence[HarnessQuestion]) -> Iterable[Generation]: ...
