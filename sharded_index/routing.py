"""Query routing: the ordered list of shards to probe for a query."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import log

import pandas as pd

from sharded_index.data.corpus import Corpus
from sharded_index.graph.cooccurrence import node_strength
from sharded_index.partition.model import TermPartition
from sharded_index.text import Tokenizer


def cover_weights(kind: str, corpus: Corpus, edges: pd.DataFrame) -> dict[str, float]:
    """Term weights that order the probes of a query.

    ``idf`` is the BM25 idf of Whoosh, ``ln(N / (df + 1)) + 1``: the first
    probed shard holds the query terms that weigh most in the ranking.
    ``strength`` is the node strength in the graph (0 for out-of-graph terms).
    """
    if kind == "idf":
        n_docs = corpus.n_docs
        return {
            term: log(n_docs / (df + 1)) + 1.0
            for term, df in zip(corpus.vocabulary, corpus.df.tolist(), strict=True)
        }
    if kind == "strength":
        return node_strength(edges)
    msg = f"unknown cover weight: {kind!r}"
    raise ValueError(msg)


def greedy_cover(
    candidates: Mapping[str, tuple[int, ...]],
    weights: Mapping[str, float],
    max_shards: int | None = None,
) -> list[int]:
    """Greedy weighted set cover of the terms by shards, in probing order.

    Each step takes the shard whose still-uncovered terms weigh most; ties
    go to the smaller shard id.
    """
    uncovered = set(candidates)
    order: list[int] = []
    while uncovered and (max_shards is None or len(order) < max_shards):
        gains: dict[int, float] = defaultdict(float)
        for term in sorted(uncovered):
            for shard in candidates[term]:
                gains[shard] += weights.get(term, 0.0)
        best = min(gains, key=lambda shard: (-gains[shard], shard))
        order.append(best)
        uncovered = {term for term in uncovered if best not in candidates[term]}
    return order


@dataclass(frozen=True)
class Router:
    """Routes queries over a partition."""

    partition: TermPartition
    weights: Mapping[str, float]
    tokenizer: Tokenizer

    def terms(self, query: str) -> set[str]:
        """Distinct terms of a query."""
        return set(self.tokenizer(query))

    def known(self, terms: Iterable[str]) -> dict[str, tuple[int, ...]]:
        """The terms that have a shard, with their shards; the rest match no document."""
        term_shards = self.partition.term_shards
        return {term: term_shards[term] for term in terms if term in term_shards}

    def cover(self, query: str, max_shards: int | None = None) -> list[int]:
        """Shards that together hold a shard of every query term, most important first."""
        return greedy_cover(self.known(self.terms(query)), self.weights, max_shards)
