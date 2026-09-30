"""The judge port and its result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .taxonomy import FailureMode


class JudgeStatus(StrEnum):
    """Why a judgement does or does not exist.

    UNSCORED is a first-class outcome, not an error: a judge that cannot cite its
    evidence is excluded from the agreement statistics rather than being given the
    benefit of the doubt.
    """

    OK = "ok"
    UNPARSEABLE = "unparseable"
    OUT_OF_TAXONOMY = "out_of_taxonomy"
    UNCITED = "uncited"
    API_ERROR = "api_error"


class Verdict(StrEnum):
    """The judge's answer to "is the generated query correct?" — never the metric.

    `SQL_OK` exists so the judge can be asked about queries that are *correct*, which is
    what makes the audit sample of successes meaningful. Without it the judge could only
    ever return "wrong", and its pass rate on failures would be unfalsifiable.
    """

    SQL_OK = "sql_ok"
    GENERATOR_ERROR = "generator_error"
    SQL_ERROR = "sql_error"


@dataclass(frozen=True)
class Judgement:
    question_id: str
    db_id: str
    status: JudgeStatus
    verdict: Verdict | None = None
    failure_mode: FailureMode | None = None
    confidence: float | None = None
    rationale: str = ""
    evidence_span: str = ""
    model: str = ""
    prompt_version: str = ""
    cache_key: str = ""
    attempts: int = 1
    error: str | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.status is JudgeStatus.OK

    def to_row(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "db_id": self.db_id,
            "status": str(self.status),
            "verdict": str(self.verdict) if self.verdict else None,
            "failure_mode": str(self.failure_mode) if self.failure_mode else None,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "evidence_span": self.evidence_span,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "cache_key": self.cache_key,
            "attempts": self.attempts,
            "error": self.error,
            "extras": self.extras,
        }

    @classmethod
    def from_row(cls, row: dict) -> Judgement:
        mode = row.get("failure_mode")
        verdict = row.get("verdict")
        return cls(
            question_id=row["question_id"],
            db_id=row["db_id"],
            status=JudgeStatus(row["status"]),
            verdict=Verdict(verdict) if verdict else None,
            failure_mode=FailureMode(mode) if mode else None,
            confidence=row.get("confidence"),
            rationale=row.get("rationale", ""),
            evidence_span=row.get("evidence_span", ""),
            model=row.get("model", ""),
            prompt_version=row.get("prompt_version", ""),
            cache_key=row.get("cache_key", ""),
            attempts=int(row.get("attempts") or 1),
            error=row.get("error"),
            extras=row.get("extras") or {},
        )


class JudgeClient(Protocol):
    """Minimal chat surface, so the graph does not know which transport it is using."""

    model: str

    def complete(self, messages: list[dict[str, str]], **options: object) -> str: ...
