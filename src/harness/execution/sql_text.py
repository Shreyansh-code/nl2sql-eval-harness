"""Lexical helpers for SQL text. Deliberately lexical, not a parser.

These functions answer questions we must get right *before* handing text to SQLite:
is this a single statement, is it read-only, does the gold query demand ordering.
A full parser would be more precise, but the failure mode we care about is a
model emitting a destructive or multi-statement string, and SQLite's own
authorizer plus the read-only connection is the real backstop (see guard.py).
"""

from __future__ import annotations

import re

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_IDENT_LITERAL = re.compile(r'"(?:[^"]|"")*"')
_TRAILING = re.compile(r"[;\s]+$")


def mask_literals(sql: str) -> str:
    """Replace string/identifier literals and comments with spaces of equal length.

    Keeps byte offsets stable so a regex match index still points at the original text.
    """
    masked = sql
    for pattern in (_BLOCK_COMMENT, _LINE_COMMENT, _STRING_LITERAL, _IDENT_LITERAL):
        masked = pattern.sub(lambda m: " " * len(m.group(0)), masked)
    return masked


def strip_literals(sql: str) -> str:
    """Masked and collapsed to single spaces. For keyword detection only."""
    return re.sub(r"\s+", " ", mask_literals(sql)).strip()


def is_single_statement(sql: str) -> bool:
    """True when the text holds exactly one statement.

    A trailing semicolon is fine. `SELECT 1; DROP TABLE t` is not.
    """
    masked = strip_literals(sql)
    body = _TRAILING.sub("", masked)
    return ";" not in body


def leading_keyword(sql: str) -> str:
    """Uppercased first word, ignoring comments and leading punctuation."""
    cleaned = strip_literals(sql).lstrip("( \t\n")
    match = re.match(r"[A-Za-z_]+", cleaned)
    return match.group(0).upper() if match else ""


READ_ONLY_LEADERS = frozenset({"SELECT", "WITH", "VALUES", "EXPLAIN"})

FORBIDDEN_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "REPLACE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "REINDEX",
    "TRIGGER",
)

_KEYWORD_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)

_ORDER_BY_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def forbidden_keywords(sql: str) -> list[str]:
    """Forbidden keywords appearing as whole words outside literals and comments.

    Catches `CREATE TABLE ... AS SELECT` and `INSERT INTO ... SELECT` even when the
    statement legally starts with SELECT/WITH. `REPLACE(...)` is exempt because it is
    a scalar string function, not `REPLACE INTO`.
    """
    masked = mask_literals(sql)
    found = set()
    for match in _KEYWORD_RE.finditer(masked):
        keyword = match.group(0).upper()
        if keyword == "REPLACE":
            tail = masked[match.end() :]
            if re.match(r"\s*\(", tail):
                continue
        found.add(keyword)
    return sorted(found)


def has_order_by(sql: str) -> bool:
    """Whether the query demands a specific row order.

    Only consulted on the *gold* query: if gold orders, the generated query's
    ordering is part of correctness and comparison becomes order-sensitive.
    """
    return bool(_ORDER_BY_RE.search(mask_literals(sql)))


def first_code_offset(sql: str) -> int:
    """Index of the first non-literal, non-comment character."""
    masked = mask_literals(sql)
    for index, char in enumerate(masked):
        if char not in " \t\n\r":
            return index
    return len(sql)
