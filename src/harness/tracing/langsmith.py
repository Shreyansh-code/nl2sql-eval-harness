"""LangSmith tracing, opt-in and off by default.

Two reasons it is a separate module with a single entry point: tracing must be
impossible to leave on by accident (a run has to be reproducible without an account),
and every call site should not have to know whether tracing exists. When it is off,
`trace` yields the payload untouched and costs one boolean check.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_CLIENT: Any | None = None
_ENABLED = False


def configure(enabled: bool) -> bool:
    """Turn tracing on or off for this process. Returns the effective state."""
    global _CLIENT, _ENABLED
    if not enabled:
        _CLIENT, _ENABLED = None, False
        return False
    try:
        from langsmith import Client
    except ImportError:
        return False
    if not os.environ.get("LANGCHAIN_API_KEY") and not os.environ.get("LANGSMITH_API_KEY"):
        return False
    _CLIENT = Client()
    _ENABLED = True
    return True


def is_enabled() -> bool:
    return _ENABLED


@contextmanager
def trace(name: str, inputs: dict[str, Any] | None = None, **metadata: Any) -> Iterator[dict]:
    """Wrap a unit of work in a LangSmith run when tracing is on.

    Yields a mutable outputs dict; whatever is in it when the block exits is recorded.
    Exceptions are propagated and recorded, because a trace that hides a failed call is
    worse than no trace.
    """
    if not _ENABLED or _CLIENT is None:
        outputs: dict[str, Any] = {}
        yield outputs
        return

    run = _CLIENT.create_run(
        name=name,
        inputs=_redact(inputs or {}),
        metadata=metadata,
        project_name=os.environ.get("LANGCHAIN_PROJECT") or "nl2sql-eval-harness",
    )
    outputs = {}
    try:
        yield outputs
    except Exception as exc:
        _CLIENT.update_run(run.id, error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        _CLIENT.update_run(run.id, outputs=_redact(outputs), end_time=_client_now())


_SECRET_MARKERS = ("key", "token", "secret", "password", "authorization")


def _redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Never let a credential reach a trace store."""
    return {key: ("[redacted]" if _is_secret(key) else value) for key, value in payload.items()}


def _is_secret(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SECRET_MARKERS)


def _client_now() -> Any:
    from datetime import UTC, datetime

    return datetime.now(UTC)
