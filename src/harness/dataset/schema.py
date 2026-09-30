"""Read a BIRD SQLite file's shape straight from the file.

No hand-written schema, no database_description CSVs: BIRD databases are small enough
that the DDL plus a bounded sample of low-cardinality columns is both cheaper and more
honest than a curated description.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

MAX_SAMPLE_VALUES = 8
MAX_SAMPLE_COLUMNS_PER_TABLE = 12
MAX_TABLES = 30


@dataclass(frozen=True)
class Introspection:
    tables: tuple[str, ...]
    ddl: str
    column_values: dict[str, dict[str, list[object]]]


def _readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _is_internal(name: str) -> bool:
    return name.startswith("sqlite_")


def introspect(db_path: Path) -> Introspection:
    """Collect table DDL and distinct-value samples for low-cardinality text columns."""
    conn = _readonly(db_path)
    try:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            if not _is_internal(row[0])
        ]
        tables = tuple(names[:MAX_TABLES])

        ddl_parts = []
        for table in tables:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?", (table,)
            ).fetchone()
            if row and row[0]:
                ddl_parts.append(f"{row[0].strip()};")
            column_rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            columns = [c[1] for c in column_rows][:MAX_SAMPLE_COLUMNS_PER_TABLE]
            if columns:
                ddl_parts.append(f"-- columns of {table}: " + ", ".join(str(c) for c in columns))

        column_values = {table: _sample_columns(conn, table) for table in tables}
        return Introspection(tables=tables, ddl="\n".join(ddl_parts), column_values=column_values)
    finally:
        conn.close()


def _sample_columns(conn: sqlite3.Connection, table: str) -> dict[str, list[object]]:
    """Distinct values for text columns, so a human can decode coded fields.

    Sampled per column with a LIMIT rather than COUNTing the whole table: BIRD tables
    are large and the distinct-value cap is what we actually need.
    """
    column_rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    samples: dict[str, list[object]] = {}
    for column_row in column_rows:
        column = column_row[1]
        declared_type = (column_row[2] or "").upper()
        quoted = f'"{column}"'
        try:
            if "INT" in declared_type:
                continue
            rows = conn.execute(
                f'SELECT DISTINCT {quoted} FROM "{table}" '
                f"WHERE {quoted} IS NOT NULL LIMIT {MAX_SAMPLE_VALUES + 1}"
            ).fetchall()
        except sqlite3.Error:
            continue
        values = [r[0] for r in rows]
        # Skip high-cardinality columns: more than MAX_SAMPLE_VALUES distinct values
        # means the column is an identifier or free text, and its sample is noise.
        if 0 < len(values) <= MAX_SAMPLE_VALUES:
            samples[column] = values
    return samples
