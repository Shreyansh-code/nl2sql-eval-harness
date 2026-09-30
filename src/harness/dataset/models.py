"""The one record type the rest of the pipeline passes around."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SchemaPayload:
    db_id: str
    tables: tuple[str, ...]
    ddl: str
    column_values: dict[str, dict[str, list[object]]] = field(default_factory=dict)

    def to_prompt_block(self, *, value_sample_limit: int = 3) -> str:
        """Render the schema for the generator prompt.

        Column values are the reason this exists rather than raw DDL: on BIRD the
        question frequently turns on a coded column ("M" = male) whose meaning is only
        discoverable by looking at the data.
        """
        sections = [f"-- database: {self.db_id}", self.ddl.strip()]
        for table, columns in self.column_values.items():
            if not columns:
                continue
            lines = [f"-- sample values from {table}:"]
            for column, values in columns.items():
                shown = ", ".join(repr(v) for v in list(values)[:value_sample_limit])
                lines.append(f"--   {table}.{column} in ({shown})")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)


@dataclass(frozen=True)
class HarnessQuestion:
    question_id: str
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str
    db_path: Path
    schema: SchemaPayload

    @property
    def schema_hash_inputs(self) -> str:
        """Inputs for the generation cache key. Schema identity, not just db id."""
        return f"{self.db_id}:{self.schema.ddl}"
