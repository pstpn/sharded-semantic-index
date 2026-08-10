"""The experiment dataset: MS MARCO v2.1 extraction and views over the pairs."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from .text import normalize_text


def docs_from_pairs(pairs: pd.DataFrame) -> dict[str, str]:
    """``doc_id → doc_text`` (pairs already deduplicate docs by MD5 id)."""
    return dict(zip(pairs["doc_id"], pairs["doc_text"]))


def queries_from_pairs(pairs: pd.DataFrame) -> list[str]:
    """Unique queries in first-appearance order — deterministic across runs."""
    return list(dict.fromkeys(pairs["query"]))


def split_queries(pairs: pd.DataFrame, train_ratio: float) -> tuple[list[str], list[str]]:
    """Deterministic train/holdout split of unique queries.

    The NPMI graph is built on the train part only; recall and fanout are
    measured on the holdout — otherwise the partition would be fit on the
    very queries it is evaluated with.
    """
    queries = queries_from_pairs(pairs)
    n_train = int(len(queries) * train_ratio)
    return queries[:n_train], queries[n_train:]


def _row_passages(row: dict[str, Any], *, selected_only: bool) -> list[str]:
    """Pick passage texts from one MS MARCO row.

    Rows carry several candidate passages plus ``is_selected`` flags marking
    the ones a human judged relevant.

    - Flags present, ``selected_only=True``: only the selected passages
      (possibly none — the caller then skips the row).
    - Flags present, ``selected_only=False``: selected passages, falling back
      to all passages when nothing is selected.
    - No flags: all non-empty passages.
    """
    passages = row.get("passages") or {}
    texts = [str(t) for t in passages.get("passage_text") or []]
    flags = list(passages.get("is_selected") or [])

    if flags:
        if selected_only:
            return [t for t, f in zip(texts, flags) if int(f) == 1]
        selected = [t for t, f in zip(texts, flags) if int(f) == 1 and t.strip()]
        if selected:
            return selected

    return [t for t in texts if t.strip()]


def extract_query_doc_pairs(
    split,
    *,
    max_rows: int = 200_000,
    max_passages_per_query: int = 3,
    selected_only: bool = True,
) -> list[dict[str, str]]:
    """Extract normalized (query, document) pairs from MS MARCO v2.1.

    Both queries and passages go through :func:`normalize_text`, and
    ``doc_id`` is the MD5 of the normalized text — identical passages
    deduplicate into one document.

    Parameters
    ----------
    split:
        HuggingFace dataset split (e.g. ``ds["train"]``).
    max_rows:
        Number of dataset rows to scan.
    max_passages_per_query:
        Cap on passages taken per query.
    selected_only:
        Keep only passages marked relevant (rows without any are skipped).

    Returns
    -------
    list of dicts
        Each with keys ``query``, ``doc_id``, ``doc_text``.
    """
    pairs: list[dict[str, str]] = []

    for i in range(min(len(split), max_rows)):
        row = split[i]
        query = normalize_text(row.get("query", ""))
        if not query:
            continue

        for passage in _row_passages(row, selected_only=selected_only)[:max_passages_per_query]:
            doc_text = normalize_text(passage)
            if not doc_text:
                continue
            pairs.append({
                "query": query,
                "doc_id": hashlib.md5(doc_text.encode("utf-8")).hexdigest(),
                "doc_text": doc_text,
            })

    return pairs
