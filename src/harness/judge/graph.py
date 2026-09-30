"""The judge as a small LangGraph state machine.

Four nodes, one conditional edge, and a terminal `unscored` state. The graph earns its
place on exactly one behaviour: a bounded, defect-naming retry followed by a degrade.
Written as nested functions that behaviour is easy to get subtly wrong — a retry that
loops, a fallback that silently accepts a bad label, a second call that re-sends the whole
context for no reason. Making the states explicit means `unscored` is a *designed*
outcome that the metrics can count, not an exception that escapes.

    prepare ──> judge ──> validate ──┬── ok ──────────> persist
                    ▲                 │
                    └── retry (once, on a named defect)
                                      │
                                      └── still bad ──> unscored
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..pipeline import ScoredQuestion
from ..tracing import langsmith
from .port import JudgeClient, Judgement, JudgeStatus
from .prompts import PROMPT_VERSION, build_messages, build_retry_messages, build_user_prompt
from .validate import to_judgement, validate

MAX_ATTEMPTS = 2


class JudgeState(TypedDict, total=False):
    item: ScoredQuestion
    messages: list[dict[str, str]]
    context: str
    response: str
    attempts: int
    complaint: str | None
    judgement: Judgement | None
    error: str | None


def cache_key(model: str, prompt_version: str, item: ScoredQuestion) -> str:
    """Content-addressed so a re-judge of the same inputs is free."""
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": prompt_version,
            "question_id": item.question.question_id,
            "generated_sql": item.generation.sql,
            "gold_sql": item.question.gold_sql,
            "verdict_of_record": item.match.verdict,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass
class JudgeGraph:
    """Compiled once and reused across questions.

    `judge_cache` maps cache key -> Judgement so that a second pass over the same run, or
    a re-run of the CLI, does not re-pay for judgements it already has.
    """

    client: JudgeClient
    model: str
    prompt_version: str = PROMPT_VERSION
    max_attempts: int = MAX_ATTEMPTS
    judge_cache: dict[str, Judgement] = field(default_factory=dict)
    _compiled: Any = field(default=None, init=False, repr=False)

    def compiled(self) -> Any:
        if self._compiled is None:
            self._compiled = self._build().compile()
        return self._compiled

    def judge(self, item: ScoredQuestion) -> Judgement:
        key = cache_key(self.model, self.prompt_version, item)
        cached = self.judge_cache.get(key)
        if cached is not None:
            return cached

        state: JudgeState = {"item": item, "attempts": 0, "error": None}
        with langsmith.trace(
            "judge_failure",
            inputs={
                "question_id": item.question.question_id,
                "db_id": item.question.db_id,
                "correct": item.correct,
            },
            model=self.model,
            prompt_version=self.prompt_version,
        ) as span:
            final: JudgeState = self.compiled().invoke(state)
            judgement = final.get("judgement")
            if judgement is None:
                judgement = self._unscored(item, final.get("error") or "judge produced no result")
            span["status"] = str(judgement.status)
            span["failure_mode"] = str(judgement.failure_mode) if judgement.failure_mode else None
            span["verdict"] = str(judgement.verdict) if judgement.verdict else None
            span["attempts"] = judgement.attempts

        self.judge_cache[key] = judgement
        return judgement

    # --- nodes -----------------------------------------------------------------

    def _node_prepare(self, state: JudgeState) -> JudgeState:
        item = state["item"]
        return {
            "context": build_user_prompt(item),
            "messages": build_messages(item),
            "attempts": 0,
            "complaint": None,
        }

    def _node_judge(self, state: JudgeState) -> JudgeState:
        """Call the model. A transport failure is a retryable defect, not a crash.

        One question's 500 must not abort a 60-item judging pass, so the exception is
        turned into state and the graph either retries once or degrades to `unscored`.
        """
        try:
            response = self.client.complete(state["messages"])
        except Exception as exc:  # noqa: BLE001 - any transport error is retryable here
            return {
                "response": "",
                "attempts": state.get("attempts", 0) + 1,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {"response": response, "attempts": state.get("attempts", 0) + 1, "error": None}

    def _node_validate(self, state: JudgeState) -> JudgeState:
        if state.get("error"):
            # A transport failure produces no response to validate; carry the error so the
            # retry names it rather than complaining about empty JSON.
            return {
                "judgement": None,
                "complaint": f"request failed: {state['error']}",
            }
        item = state["item"]
        validated = validate(state.get("response", ""), state.get("context", ""))
        judgement = to_judgement(
            validated,
            question_id=item.question.question_id,
            db_id=item.question.db_id,
            model=self.model,
            prompt_version=self.prompt_version,
            cache_key=cache_key(self.model, self.prompt_version, item),
            attempts=state.get("attempts", 1),
        )
        return {"judgement": judgement, "complaint": validated.complaint}

    def _node_retry(self, state: JudgeState) -> JudgeState:
        """One constrained retry that names the defect, then give up."""
        item = state["item"]
        complaint = state.get("complaint") or state.get("error") or "response was rejected"
        if state.get("attempts", 0) >= self.max_attempts:
            return {"judgement": None, "error": complaint}
        return {"messages": build_retry_messages(item, complaint)}

    def _node_persist(self, state: JudgeState) -> JudgeState:
        return {}

    # --- graph -----------------------------------------------------------------

    def _route(self, state: JudgeState) -> str:
        judgement = state.get("judgement")
        if judgement is not None and judgement.status is JudgeStatus.OK:
            return "persist"
        if state.get("attempts", 0) >= self.max_attempts:
            return "unscored"
        return "retry"

    def _build(self) -> StateGraph:
        graph = StateGraph(JudgeState)
        graph.add_node("prepare", self._node_prepare)
        graph.add_node("judge", self._node_judge)
        graph.add_node("validate", self._node_validate)
        graph.add_node("retry", self._node_retry)
        graph.add_node("persist", self._node_persist)
        graph.add_node("unscored", self._node_retry)

        graph.set_entry_point("prepare")
        graph.add_edge("prepare", "judge")
        graph.add_edge("judge", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route,
            {"persist": "persist", "retry": "retry", "unscored": "unscored"},
        )
        graph.add_edge("retry", "judge")
        graph.add_edge("persist", END)
        graph.add_edge("unscored", END)
        return graph

    def _unscored(self, item: ScoredQuestion, error: str) -> Judgement:
        return Judgement(
            question_id=item.question.question_id,
            db_id=item.question.db_id,
            status=JudgeStatus.UNPARSEABLE,
            model=self.model,
            prompt_version=self.prompt_version,
            cache_key=cache_key(self.model, self.prompt_version, item),
            error=error,
        )
