"""Validate a judge response.

A judge that returns prose, invents a category, or cites text that is not in front of it
is not a measurement instrument, it is a text generator. Everything here exists to make
those failures visible and to *exclude* them from the agreement statistics rather than
absorb them.

Three checks, in order of how much they matter:

1. the label is in the taxonomy (no silent nearest-bucket fallback);
2. `evidence_span` occurs verbatim in the context the judge was given;
3. the object is well-formed JSON with a bounded confidence and a short rationale.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .port import Judgement, JudgeStatus, Verdict
from .taxonomy import FailureMode, coerce

MAX_RATIONALE_SENTENCES = 4
MIN_SPAN_LENGTH = 3
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class Validated:
    status: JudgeStatus
    verdict: Verdict | None = None
    failure_mode: FailureMode | None = None
    confidence: float | None = None
    rationale: str = ""
    evidence_span: str = ""
    complaint: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is JudgeStatus.OK


def validate(response: str, context: str) -> Validated:
    """Parse and check one judge response against the context it was given."""
    payload = _extract_json(response)
    if payload is None:
        return Validated(JudgeStatus.UNPARSEABLE, complaint="response was not a JSON object")

    if "verdict" not in payload or "failure_mode" not in payload:
        return Validated(
            JudgeStatus.UNPARSEABLE, complaint="missing required fields verdict/failure_mode"
        )

    verdict = _parse_verdict(payload.get("verdict"))
    if verdict is None:
        return Validated(
            JudgeStatus.UNPARSEABLE,
            complaint="verdict must be one of sql_ok, generator_error, sql_error",
        )

    raw_mode = payload.get("failure_mode")
    mode_text = raw_mode if isinstance(raw_mode, str) else None

    # A correct query has no failure mode. Allowing both at once would let a judge hedge,
    # and the hedge would then be counted in whichever statistic flattered it.
    if verdict is Verdict.SQL_OK:
        if (mode_text or "").strip().upper() not in {"NONE", ""}:
            return Validated(
                JudgeStatus.OUT_OF_TAXONOMY,
                complaint="verdict sql_ok must be paired with failure_mode NONE",
            )
        return _check_common(payload, Verdict.SQL_OK, None, context)

    mode = coerce(mode_text)
    if mode is None:
        # Deliberately not coerced to a default: an invented category is a defect we
        # want counted, not smoothed away.
        return Validated(
            JudgeStatus.OUT_OF_TAXONOMY,
            complaint=f"failure_mode {raw_mode!r} is not one of the eleven codes",
        )

    return _check_common(payload, verdict, mode, context)


def _check_common(
    payload: dict, verdict: Verdict, mode: FailureMode | None, context: str
) -> Validated:
    """Checks that apply whatever the verdict: a citation, a number, a reason."""
    span = payload.get("evidence_span")
    if not isinstance(span, str) or not span.strip():
        return Validated(JudgeStatus.UNCITED, complaint="evidence_span was empty")
    span = span.strip()
    if len(span) < MIN_SPAN_LENGTH:
        return Validated(
            JudgeStatus.UNCITED, complaint="evidence_span is too short to be a citation"
        )
    if span not in context:
        return Validated(
            JudgeStatus.UNCITED,
            complaint="evidence_span is not a verbatim substring of the supplied context",
        )

    confidence = _parse_confidence(payload.get("confidence"))
    if confidence is None:
        return Validated(JudgeStatus.UNPARSEABLE, complaint="confidence must be a number 0-1")

    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return Validated(JudgeStatus.UNPARSEABLE, complaint="rationale was empty")

    return Validated(
        JudgeStatus.OK,
        verdict=verdict,
        failure_mode=mode,
        confidence=confidence,
        rationale=_shorten(rationale),
        evidence_span=span,
    )


def to_judgement(
    validated: Validated,
    *,
    question_id: str,
    db_id: str,
    model: str,
    prompt_version: str,
    cache_key: str,
    attempts: int = 1,
) -> Judgement:
    return Judgement(
        question_id=question_id,
        db_id=db_id,
        status=validated.status,
        verdict=validated.verdict,
        failure_mode=validated.failure_mode,
        confidence=validated.confidence,
        rationale=validated.rationale,
        evidence_span=validated.evidence_span,
        model=model,
        prompt_version=prompt_version,
        cache_key=cache_key,
        attempts=attempts,
        error=validated.complaint,
    )


def _extract_json(response: str) -> dict | None:
    if not response or not response.strip():
        return None
    candidates: list[str] = []
    fenced = _FENCE_RE.findall(response)
    candidates.extend(block.strip() for block in fenced)
    match = _OBJECT_RE.search(response)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_verdict(value: object) -> Verdict | None:
    if not isinstance(value, str):
        return None
    try:
        return Verdict(value.strip().lower())
    except ValueError:
        return None


def _parse_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not 0.0 <= number <= 1.0:
        return None
    return number


def _shorten(rationale: str, limit: int = MAX_RATIONALE_SENTENCES) -> str:
    """Collapse whitespace and cap length. The cap is a prompt promise, not a hope."""
    text = " ".join(rationale.split())
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    return " ".join(sentences[:limit])
