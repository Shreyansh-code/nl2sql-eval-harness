"""Standard-mode generator: one request per question, bounded client-side concurrency.

Uses the raw HTTP API through `urllib` rather than the SDK on purpose — the project
deliberately depends on almost nothing, and the batch adapter needs the same wire format
anyway, so the two adapters can share one request builder and one response parser.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..config import Settings
from ..dataset.models import HarnessQuestion
from ..tracing import langsmith
from .cache import GenerationCache, cache_key
from .parsing import parse_sql
from .port import Generation, GenerationStatus
from .prompts import PROMPT_VERSION, build_messages

API_ROOT = "https://api.openai.com/v1"
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = (1.0, 3.0, 8.0)


@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str | None


class StandardGenerator:
    """Implements the SqlGenerator port against /v1/chat/completions."""

    def __init__(self, settings: Settings, cache: GenerationCache | None = None) -> None:
        if not settings.generator_model:
            raise ValueError("GENERATOR_MODEL is not set")
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.settings = settings
        self.model = settings.generator_model
        self.prompt_version = PROMPT_VERSION
        self.cache = cache if cache is not None else GenerationCache()
        self._endpoint = f"{(settings.openai_base_url or API_ROOT).rstrip('/')}/chat/completions"

    def generate(self, questions: Sequence[HarnessQuestion]) -> list[Generation]:
        """Generate for every question, reusing cached results and writing new ones.

        Cache hits keep their original latency and token counts so a mixed run's cost
        figures still mean something.
        """
        pending: list[HarnessQuestion] = []
        results: dict[str, Generation] = {}

        for question in questions:
            key = self.key_for(question)
            cached = self.cache.get(key)
            if cached is not None:
                results[question.question_id] = cached
            else:
                pending.append(question)

        if not pending:
            return [results[q.question_id] for q in questions]

        workers = max(1, self.settings.max_concurrency)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            produced = pool.map(self._generate_one, pending)

        for generation in produced:
            self.cache.put(generation)
            results[generation.question_id] = generation

        return [results[q.question_id] for q in questions]

    def key_for(self, question: HarnessQuestion) -> str:
        return cache_key(
            model=self.model,
            prompt_version=self.prompt_version,
            question_id=question.question_id,
            schema_ddl=question.schema.ddl,
            question=question.question,
            evidence=question.evidence,
        )

    def _generate_one(self, question: HarnessQuestion) -> Generation:
        key = self.key_for(question)
        messages = build_messages(question)

        with langsmith.trace(
            "generate_sql",
            inputs={
                "question_id": question.question_id,
                "db_id": question.db_id,
                "question": question.question,
                "evidence": question.evidence,
            },
            model=self.model,
            prompt_version=self.prompt_version,
            cache_key=key,
        ) as span:
            started = time.monotonic()
            result, error = self._chat(messages)
            latency_ms = int((time.monotonic() - started) * 1000)
            if result is None:
                span["status"] = str(GenerationStatus.API_ERROR)
                span["error"] = error
                return Generation(
                    question_id=question.question_id,
                    db_id=question.db_id,
                    status=GenerationStatus.API_ERROR,
                    sql=None,
                    raw_response="",
                    model=self.model,
                    prompt_version=self.prompt_version,
                    cache_key=key,
                    error=error,
                )

            parsed = parse_sql(result.text)
            status = (
                GenerationStatus.OK
                if parsed.ok
                else (
                    GenerationStatus.EMPTY
                    if parsed.reason == "empty_response"
                    else GenerationStatus.UNPARSEABLE
                )
            )
            generation = Generation(
                question_id=question.question_id,
                db_id=question.db_id,
                status=status,
                sql=parsed.sql,
                raw_response=result.text,
                model=result.model or self.model,
                prompt_version=self.prompt_version,
                cache_key=key,
                latency_ms=latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                error=parsed.reason,
                extras={"blocks_seen": parsed.blocks_seen, "finish_reason": result.finish_reason},
            )
            span["status"] = str(status)
            span["sql"] = parsed.sql
            span["parse_reason"] = parsed.reason
            span["output_tokens"] = result.output_tokens
            return generation

    def _request_body(self, messages: list[dict[str, str]]) -> dict[str, object]:
        """Build the request payload.

        `temperature` is included only when configured. Reasoning models such as
        gpt-6-luna reject temperature=0 outright and accept only their default, so
        sending one unconditionally turns every request into a 400.
        """
        body: dict[str, object] = {"model": self.model, "messages": messages}
        if self.settings.generator_temperature is not None:
            body["temperature"] = self.settings.generator_temperature
        return body

    def _chat(self, messages: list[dict[str, str]]) -> tuple[ChatResult | None, str | None]:
        """POST the request with bounded retries on transient failures."""
        payload = json.dumps(self._request_body(messages)).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        last_error: str | None = None
        for attempt in range(MAX_ATTEMPTS):
            request = urllib.request.Request(
                self._endpoint, data=payload, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return _parse_chat(body), None
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                last_error = f"http_{exc.code}: {detail}"
                if exc.code not in RETRYABLE_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < len(BACKOFF_SECONDS):
                time.sleep(BACKOFF_SECONDS[attempt])

        return None, last_error or "unknown_error"


def _parse_chat(body: dict) -> ChatResult:
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    return ChatResult(
        text=message.get("content") or "",
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        model=body.get("model") or "",
        finish_reason=choice.get("finish_reason"),
    )
