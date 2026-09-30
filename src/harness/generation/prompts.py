"""The generator prompt.

Two decisions worth stating, because both are measurable and both are the kind of thing
that gets a benchmark result dismissed:

* **Schema is introspected, not described.** The DDL plus distinct values for
  low-cardinality columns is what makes BIRD's coded fields usable at all — `position`
  holding `'M'` is the difference between a right filter and a hallucinated column.
* **`evidence` is included.** BIRD ships it, and hiding it would be measuring a
  different (easier, less realistic) task. The README must say the evidence field is
  available, because an unlabelled 60% EX number is not comparable to a published one.
"""

from __future__ import annotations

from ..dataset.models import HarnessQuestion

PROMPT_VERSION = "v1"

SYSTEM = """You write SQLite queries for a specific database.

You are given the database schema, a natural-language question, and sometimes an
`evidence` hint that disambiguates how the question's terms map onto the schema.

Rules:
- Return exactly one SQLite statement, inside a single ```sql code block.
- No prose before or after the block. No explanation, no comments.
- Use only tables and columns present in the schema.
- Prefer explicit JOIN ... ON over comma joins.
- Do not add an ORDER BY unless the question asks for a specific order.
"""


def build_user_prompt(question: HarnessQuestion) -> str:
    parts = [
        "## Schema",
        question.schema.to_prompt_block(),
        "",
        "## Question",
        question.question.strip(),
    ]
    if question.evidence.strip():
        parts += ["", "## Evidence", question.evidence.strip()]
    parts += [
        "",
        "## Answer",
        "Return one ```sql block containing the query.",
    ]
    return "\n".join(parts)


def build_messages(question: HarnessQuestion) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt(question)},
    ]
