"""Text normalization and tokenization shared by indexing, routing and the graph."""

from __future__ import annotations

import re
from dataclasses import dataclass

WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)  # noqa: RUF001
MIN_TOKEN_LEN = 2
"""Whoosh's ``StopFilter`` drops shorter tokens, so they are never searchable."""


def normalize_text(text: str) -> str:
    """Lowercase and keep alphanumeric runs joined by single spaces.

    Document ids hash this form and queries are deduplicated by it, so it
    must not depend on the stop list.
    """
    return " ".join(WORD_RE.findall(str(text).lower()))


@dataclass(frozen=True)
class Tokenizer:
    """Maps text to the terms that are indexed, assigned and routed."""

    stop_words: frozenset[str]

    def __call__(self, text: str) -> list[str]:
        return [
            token
            for token in WORD_RE.findall(str(text).lower())
            if len(token) >= MIN_TOKEN_LEN and token not in self.stop_words
        ]
