"""Text normalization shared by routing, indexing and graph construction.

This module is the single source of truth for tokenization: the same
lowercasing and stop-word list are used when routing queries/documents to
shards (:func:`tokenize`), when indexing documents in Whoosh
(:data:`~sharded_index.indexing.WHOOSH_ANALYZER`) and when building the term
co-occurrence graph (:func:`~sharded_index.graph.build_npmi_graph`).
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS: frozenset[str] = frozenset(ENGLISH_STOP_WORDS)
"""English stop words — dropped from queries, documents, the NPMI graph and Whoosh indices."""

WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
"""A token is a maximal run of latin/cyrillic letters or digits."""


def normalize_text(text: str) -> str:
    """Lowercase, keep alphanumeric tokens, drop stop words, re-join with spaces."""
    tokens = WORD_RE.findall(str(text).lower())
    return " ".join(t for t in tokens if t not in STOP_WORDS)


MIN_TOKEN_LEN = 2
"""Minimum token length — mirrors the Whoosh ``StopFilter`` default.

Whoosh never indexes single-character tokens, so routing a query by such a
term probes shards that cannot match it, and assigning documents by it
inflates duplication with unsearchable postings (the possessive fragment
``s`` alone touches 21% of the corpus).
"""


def tokenize(text: str) -> list[str]:
    """Tokenize for routing and document assignment.

    ``normalize_text(text).split()`` minus tokens shorter than
    :data:`MIN_TOKEN_LEN` — exactly the terms the Whoosh analyzer indexes,
    so a term seen at routing time is a term Whoosh indexed.
    ``normalize_text`` itself keeps short tokens: document ids are MD5 of
    the normalized text and must stay stable.
    """
    return [t for t in normalize_text(text).split() if len(t) >= MIN_TOKEN_LEN]
