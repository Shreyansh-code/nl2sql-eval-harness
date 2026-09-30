"""The judge prompt.

The judge is given the gold SQL (decided in the HLD) because it cannot say *why* a query
is wrong without knowing what right looks like. What keeps that honest is not concealment
but measurement: humans later see the same inputs, so any leniency shows up as kappa
disagreement rather than being invisible.

Two constraints are load-bearing:

* **exactly one** label from the closed taxonomy;
* **`evidence_span` must be copied verbatim** from the context supplied. This is the
  cheapest available guard against a judge that is pattern-matching rather than
  reasoning, and it is falsifiable in a way "be careful" is not.
"""

from __future__ import annotations

from ..pipeline import ScoredQuestion
from .taxonomy import describe_all

PROMPT_VERSION = "v1"

MAX_ROWS_SHOWN = 5

SYSTEM = f"""You assess a text-to-SQL model's query for a given question, and when it is
wrong you say why.

You are given the database schema, the question, an evidence hint, the model's generated
SQL, the gold SQL, and the actual execution result of both. The generated query is often
correct. Decide which case you are in before deciding why.

Return one JSON object and nothing else. No prose before or after. Schema:

{{
  "verdict": "sql_ok" | "generator_error" | "sql_error",
  "failure_mode": "<one code from the list below, or \"NONE\" when verdict is sql_ok>",
  "confidence": <number between 0 and 1>,
  "rationale": "<at most 3 sentences>",
  "evidence_span": "<a span copied VERBATIM from the context above>"
}}

- "sql_ok" means the generated query is correct: it runs and returns the right answer.
- "sql_error" means it would not run at all.
- "generator_error" means it runs but returns the wrong answer.
- "failure_mode" must be exactly one of these codes, or the string "NONE" when the
  verdict is "sql_ok":

{describe_all()}

Rules for the two fields that matter:
- "rationale" must name the specific difference that caused the wrong answer.
- "evidence_span" must be copied character-for-character from the schema, question,
  evidence, generated SQL, gold SQL, or result rows shown above. Do not paraphrase it and
  do not invent it. If you cannot quote the text your diagnosis rests on, lower
  "confidence" and quote the closest thing you did rely on.
"""


def build_user_prompt(item: ScoredQuestion) -> str:
    """Render everything the judge is allowed to see."""
    question = item.question
    parts = [
        "## Schema",
        question.schema.to_prompt_block(),
        "",
        "## Question",
        question.question.strip(),
    ]
    if question.evidence.strip():
        parts += ["", "## Evidence", question.evidence.strip()]

    parts += ["", "## Generated SQL", "```sql", (item.generation.sql or "<none>").strip(), "```"]

    if item.gold is not None:
        parts += ["", "## Gold SQL", "```sql", question.gold_sql.strip(), "```"]

    parts += [
        "",
        "## Execution of the generated SQL",
        f"outcome: {item.execution.outcome}",
    ]
    if item.execution.detail:
        parts.append(f"detail: {item.execution.detail}")
    parts.append(f"rows returned: {len(item.execution.rows)}")
    if item.execution.rows:
        parts.append("first rows:")
        for row in item.execution.rows[:MAX_ROWS_SHOWN]:
            parts.append(f"  {list(row)}")

    if item.gold is not None and item.gold.rows:
        parts += ["", "## Execution of the gold SQL", f"rows returned: {len(item.gold.rows)}"]
        parts.append("first rows:")
        for row in item.gold.rows[:MAX_ROWS_SHOWN]:
            parts.append(f"  {list(row)}")

    if item.match.first_difference:
        parts += ["", "## Observed difference"]
        parts.append(str(item.match.first_difference.get("reason")))
        for key in ("missing_from_generated", "unexpected_in_generated"):
            value = item.match.first_difference.get(key)
            if value:
                parts.append(f"{key}: {value}")

    parts += ["", "## Answer", "Return one JSON object."]
    return "\n".join(parts)


def build_messages(item: ScoredQuestion) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt(item)},
    ]


def build_retry_messages(item: ScoredQuestion, complaint: str) -> list[dict[str, str]]:
    """A single constrained retry: the original request plus the specific defect.

    Re-sending the whole conversation with a vague "try again" wastes tokens and invites
    the same defect; naming the defect is what makes the retry worth having.
    """
    return [
        *build_messages(item),
        {
            "role": "assistant",
            "content": "A previous response was rejected.",
        },
        {
            "role": "user",
            "content": (
                f"Your previous response was rejected because: {complaint}\n\n"
                "Return a corrected JSON object only. Keep every other field, and make "
                "evidence_span an exact copy of text from the context above."
            ),
        },
    ]
