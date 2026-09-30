"""Content-addressed cache for generations.

Re-runs are free, which is what makes the expensive loop in this project bearable: the
human-labeling pass re-reads judgements, and a second prompt version must re-score the
*same* generations rather than pay for them again. Keying on the content that actually
determines the output means a cache hit is safe by construction — if the schema, the
question, the evidence, the prompt, or the model changes, the key changes with it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..runs import iter_rows
from .port import Generation


def cache_key(
    *,
    model: str,
    prompt_version: str,
    question_id: str,
    schema_ddl: str,
    question: str,
    evidence: str,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": prompt_version,
            "question_id": question_id,
            "schema_ddl": schema_ddl,
            "question": question,
            "evidence": evidence,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class GenerationCache:
    """Generations already paid for, loaded from a run's generations.jsonl."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._by_key: dict[str, Generation] = {}
        if path and path.exists():
            for row in iter_rows(path):
                generation = Generation.from_row(row)
                self._by_key[generation.cache_key] = generation

    def get(self, key: str) -> Generation | None:
        return self._by_key.get(key)

    def put(self, generation: Generation) -> None:
        self._by_key[generation.cache_key] = generation

    def __contains__(self, key: str) -> bool:
        return key in self._by_key

    def __len__(self) -> int:
        return len(self._by_key)
