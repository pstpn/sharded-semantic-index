"""Per-query measurements of sharded search and static properties of a partition.

Sharded results are emulated: with collection-wide BM25 statistics the result
of probing a set of shards is the reference ranking restricted to the
documents of those shards (verified physically by the ``verify`` stage).

For a query and a budget of ``b`` shards (the first ``b`` shards of its
cover; ``full`` is the whole cover):

- ``overlap`` — share of the reference top-k found in the probed shards;
- ``volume`` — documents scanned, as a share of the corpus (a document held
  by two probed shards counts twice).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sharded_index.config import Strategy
from sharded_index.data.corpus import Corpus
from sharded_index.evaluation.membership import shard_volumes
from sharded_index.index.ranking import Rankings
from sharded_index.partition.model import TermPartition
from sharded_index.routing import Router, greedy_cover

FULL = "full"


def budget_labels(budgets: Sequence[int]) -> list[str]:
    """Budget names used in column names and tables: ``1, 2, ..., full``."""
    return [*map(str, budgets), FULL]


def measure_queries(
    router: Router,
    membership: sp.csr_matrix,
    rankings: Rankings,
    budgets: Sequence[int],
    top_k: int,
    hash_offset: int | None,
) -> pd.DataFrame:
    """One row per query of the sample, in sample order.

    Columns: ``evaluated``, ``terms`` (distinct query terms), ``known_terms``
    (those that occur in the corpus and so have a shard), ``fanout`` (shards
    in the cover), ``first_shard``, ``hash_probes`` (cover shards in the
    hash-fallback space) and ``overlap_<b>``, ``volume_<b>`` for every budget.
    Queries without a reference ranking are not evaluated: their measurements
    are NaN.
    """
    n_queries = len(rankings.queries)
    n_shards = membership.shape[1]
    sizes = shard_volumes(membership) / membership.shape[0]
    labels = budget_labels(budgets)

    n_terms = np.zeros(n_queries, dtype=np.int64)
    n_known = np.zeros(n_queries, dtype=np.int64)
    fanout = np.full(n_queries, np.nan)
    hash_probes = np.full(n_queries, np.nan)
    first_shard = np.full(n_queries, -1, dtype=np.int64)
    overlap = np.full((n_queries, len(labels)), np.nan)
    volume = np.full((n_queries, len(labels)), np.nan)

    not_probed = n_shards
    probe_order = np.full(n_shards, not_probed, dtype=np.int64)
    indptr, indices = membership.indptr, membership.indices
    for i, (query, ranking) in enumerate(zip(rankings.queries, rankings.arrays, strict=True)):
        terms = router.terms(query)
        known = router.known(terms)
        n_terms[i], n_known[i] = len(terms), len(known)
        if len(ranking) == 0:
            continue
        cover = np.asarray(greedy_cover(known, router.weights), dtype=np.int64)
        fanout[i] = len(cover)
        first_shard[i] = cover[0]
        if hash_offset is not None:
            hash_probes[i] = int((cover >= hash_offset).sum())

        probe_order[cover] = np.arange(len(cover))
        found_at = np.array(
            [probe_order[indices[indptr[doc] : indptr[doc + 1]]].min() for doc in ranking[:top_k]]
        )
        probe_order[cover] = not_probed
        for column, budget in enumerate([*budgets, len(cover)]):
            width = min(budget, len(cover))
            overlap[i, column] = (found_at < width).mean()
            volume[i, column] = sizes[cover[:width]].sum()

    frame = pd.DataFrame(
        {
            "position": np.arange(n_queries),
            "evaluated": rankings.evaluated,
            "terms": n_terms,
            "known_terms": n_known,
            "fanout": fanout,
            "first_shard": first_shard,
            "hash_probes": hash_probes,
        }
    )
    for column, label in enumerate(labels):
        frame[f"overlap_{label}"] = overlap[:, column]
        frame[f"volume_{label}"] = volume[:, column]
    return frame


def expected_hash_duplication(corpus: Corpus, n_shards: int) -> float:
    """Mean duplication of a document under random term assignment to ``n_shards`` shards.

    A document of ``m`` distinct terms occupies ``S (1 - (1 - 1/S)^m)`` shards on average.
    """
    lengths = np.asarray(corpus.doc_terms.sum(axis=1)).ravel().astype(float)
    return float((n_shards * (1.0 - (1.0 - 1.0 / n_shards) ** lengths)).mean())


def _replica_cost_shares(partition: TermPartition, corpus: Corpus) -> dict[str, float]:
    """Share of the replication cost paid by the most frequent 1% and 10% of terms.

    The cost of a replica is the document frequency of its term: that many
    documents are copied into one more shard.
    """
    df = dict(zip(corpus.vocabulary, corpus.df.tolist(), strict=True))
    cost = {
        term: df.get(term, 0) * (len(shards) - 1)
        for term, shards in partition.term_shards.items()
        if len(shards) > 1
    }
    total = sum(cost.values())
    if not total:
        return {"replica_cost_top1pct": np.nan, "replica_cost_top10pct": np.nan}
    descending = sorted(df.values(), reverse=True)

    def share(top: float) -> float:
        threshold = descending[max(int(len(descending) * top) - 1, 0)]
        return sum(value for term, value in cost.items() if df.get(term, 0) >= threshold) / total

    return {"replica_cost_top1pct": share(0.01), "replica_cost_top10pct": share(0.10)}


def partition_stats(
    partition: TermPartition,
    membership: sp.csr_matrix,
    corpus: Corpus,
    strategy: Strategy,
    hash_offset: int | None,
) -> dict[str, float]:
    """Query-independent properties of a partition: size, duplication and balance."""
    n_docs = membership.shape[0]
    volumes = np.sort(shard_volumes(membership))[::-1]
    n_shards = int((volumes > 0).sum())
    primary = partition.primary
    hash_terms = (
        np.nan
        if hash_offset is None
        else sum(1 for shard in primary.values() if shard >= hash_offset)
    )
    return {
        "shards": n_shards,
        "replication_factor": partition.replication_factor,
        "replicated_terms": sum(1 for shards in partition.term_shards.values() if len(shards) > 1),
        "duplication": membership.nnz / n_docs,
        "expected_duplication": (
            expected_hash_duplication(corpus, n_shards) if strategy.is_hash else np.nan
        ),
        "largest_shard": float(volumes[0] / n_docs),
        "top10_shards": float(volumes[:10].sum() / n_docs),
        "hash_terms": hash_terms,
        **_replica_cost_shares(partition, corpus),
    }
