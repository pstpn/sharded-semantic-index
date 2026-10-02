"""MS MARCO v2.1 extraction and views over the extracted query-passage pairs.

A dataset row is one query with about ten candidate passages, of which an
assessor marked the ones that answer it.  A row contributes its marked
passages first and then unmarked candidates (distractors) to the corpus.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from sharded_index.text import normalize_text

PAIR_COLUMNS = ["query", "doc_id", "doc_text", "is_selected"]


def _row_passages(row: Mapping[str, Any], docs_per_query: int) -> list[tuple[str, bool]]:
    """Up to ``docs_per_query`` distinct passages of a row, marked ones first.

    A row without a marked passage contributes nothing.
    """
    passages = row.get("passages") or {}
    texts = [normalize_text(text) for text in passages.get("passage_text") or []]
    flags = [int(flag) == 1 for flag in passages.get("is_selected") or []]
    if not any(flags):
        return []

    ordered = [(text, True) for text, flag in zip(texts, flags, strict=False) if flag]
    ordered += [(text, False) for text, flag in zip(texts, flags, strict=False) if not flag]

    chosen: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for text, selected in ordered:
        if text and text not in seen:
            chosen.append((text, selected))
            seen.add(text)
        if len(chosen) == docs_per_query:
            break
    return chosen


def extract_pairs(
    rows: Sequence[Mapping[str, Any]], *, max_rows: int, docs_per_query: int
) -> pd.DataFrame:
    """Extract ``(query, doc_id, doc_text, is_selected)`` rows from the dataset.

    Texts are normalized; ``doc_id`` is the MD5 of the normalized passage, so
    identical passages become one document.
    """
    pairs = []
    for i in range(min(len(rows), max_rows)):
        row = rows[i]
        query = normalize_text(row.get("query", ""))
        if not query:
            continue
        for doc_text, selected in _row_passages(row, docs_per_query):
            doc_id = hashlib.md5(doc_text.encode("utf-8")).hexdigest()
            pairs.append((query, doc_id, doc_text, selected))
    return pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def docs_from_pairs(pairs: pd.DataFrame) -> dict[str, str]:
    """``doc_id → text`` of the corpus in first-appearance order."""
    return dict(zip(pairs["doc_id"], pairs["doc_text"], strict=True))


def queries_from_pairs(pairs: pd.DataFrame) -> list[str]:
    """Unique queries in first-appearance order."""
    return list(dict.fromkeys(pairs["query"]))


def split_queries(pairs: pd.DataFrame, train_ratio: float) -> tuple[list[str], list[str]]:
    """Split the unique queries into the graph-building head and the held-out tail."""
    queries = queries_from_pairs(pairs)
    n_train = int(len(queries) * train_ratio)
    return queries[:n_train], queries[n_train:]
