"""Routing-recall evaluation of a term partition.

Methodology
-----------
The reference answer set (*ground truth*) for a query is everything a single
un-sharded BM25 index returns for it (OR over query terms, no limit).  A
partition is then scored by *routing recall*: the fraction of ground-truth
documents that live in at least one of the shards the query was routed to.
No per-shard search is needed — set intersection of the document's shards
with the probed shards is enough.

A query with an empty ground truth counts as fully recalled (1.0).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from whoosh import index as whoosh_index
from whoosh.qparser import OrGroup, QueryParser

from .partition import TermPartition
from .text import normalize_text, tokenize


def build_ground_truth(
    queries: list[str],
    full_index_dir: Path,
) -> dict[str, set[str]]:
    """All doc_ids the full (un-sharded) index returns for each query.

    This is the expensive step — one unlimited BM25 search per query.
    """
    ix = whoosh_index.open_dir(full_index_dir)
    parser = QueryParser("text", schema=ix.schema, group=OrGroup)

    ground_truth: dict[str, set[str]] = {}
    with ix.searcher() as searcher:
        for query in tqdm(queries, desc="ground truth"):
            parsed = parser.parse(normalize_text(query))
            ground_truth[query] = {hit["doc_id"] for hit in searcher.search(parsed, limit=None)}
    return ground_truth


def term_postings(doc_tokens: dict[str, Collection[str]]) -> dict[str, set[str]]:
    """Inverted map ``term → doc_ids`` over tokenized documents."""
    postings: dict[str, set[str]] = defaultdict(set)
    for doc_id, tokens in doc_tokens.items():
        for term in tokens:
            postings[term].add(doc_id)
    return dict(postings)


def build_ground_truth_fast(
    queries: list[str],
    postings: dict[str, set[str]],
) -> dict[str, set[str]]:
    """In-memory ground truth: documents sharing at least one term with the query.

    An unlimited BM25 OR-search returns a document iff it contains at least
    one query term, so with the routing tokenizer aligned to the Whoosh
    analyzer (``tokenize``; the smoke test pins the equivalence) this equals
    :func:`build_ground_truth` without touching a physical index.  Meant for
    fast method-development iterations; publication numbers use the physical
    path.
    """
    ground_truth: dict[str, set[str]] = {}
    for query in tqdm(queries, desc="ground truth (fast)"):
        matched = [postings[t] for t in set(tokenize(query)) if t in postings]
        ground_truth[query] = set().union(*matched) if matched else set()
    return ground_truth


@dataclass
class RoutingRecall:
    """Routing recall and fanout statistics of one partition."""

    recall: float
    mean_fanout: float
    median_fanout: float
    p95_fanout: float


def _recall_for_query(
    gt_ids: set[str],
    probed: set[int],
    doc_shards: dict[str, set[int]],
) -> float:
    """Fraction of ground-truth docs reachable through the probed shards."""
    if not gt_ids:
        return 1.0
    found = sum(1 for doc_id in gt_ids if doc_shards[doc_id] & probed)
    return found / len(gt_ids)


def compute_routing_recall(
    queries: list[str],
    ground_truth: dict[str, set[str]],
    partition: TermPartition,
    doc_shards: dict[str, set[int]],
) -> RoutingRecall:
    """Routing recall with full fanout (query goes to ALL shards of its terms).

    Parameters
    ----------
    queries:
        Queries to evaluate.
    ground_truth:
        ``query → doc_ids`` from :func:`build_ground_truth`.
    partition:
        Term partition that routes the queries.
    doc_shards:
        ``doc_id → shard ids`` from :func:`~sharded_index.assign_docs_to_shards`.
    """
    recalls, fanouts = [], []
    for query in queries:
        probed = partition.shards_for_query(query)
        fanouts.append(len(probed))
        recalls.append(_recall_for_query(ground_truth.get(query, set()), probed, doc_shards))

    return RoutingRecall(
        recall=float(np.mean(recalls)),
        mean_fanout=float(np.mean(fanouts)),
        median_fanout=float(np.median(fanouts)),
        p95_fanout=float(np.percentile(fanouts, 95)),
    )


def recall_by_fanout(
    queries: list[str],
    ground_truth: dict[str, set[str]],
    partition: TermPartition,
    doc_shards: dict[str, set[int]],
    *,
    max_fanout: int = 5,
) -> dict[int, float]:
    """Recall when probing only the first 1..N shards of the greedy cover.

    Shards are ordered by :meth:`cover_shards_for_query` — greedy weighted
    set cover of the query terms.  For single-assignment partitions this is
    ranking by aggregate node strength; for replicated partitions the cover
    exploits replicas to fit the query into fewer shards.

    Returns
    -------
    dict[int, float]
        ``budget → mean recall`` for budget in ``1..max_fanout``.
    """
    fanout_range = range(1, max_fanout + 1)
    recalls: dict[int, list[float]] = {f: [] for f in fanout_range}

    for query in queries:
        gt_ids = ground_truth.get(query, set())
        cover = partition.cover_shards_for_query(query)
        for f in fanout_range:
            recalls[f].append(_recall_for_query(gt_ids, set(cover[:f]), doc_shards))

    return {f: float(np.mean(recalls[f])) for f in fanout_range}


def cover_fanouts(queries: list[str], partition: TermPartition) -> list[int]:
    """Cover fanout per query: how many shards the greedy cover needs.

    This is the headline routing metric: probing the full cover always
    yields recall 1.0 (documents replicate to all shards of their terms),
    so the question is how *short* the cover is.  For single-assignment
    partitions it equals the plain fanout (one shard per distinct term
    cluster); replication exists to shrink it.
    """
    return [len(partition.cover_shards_for_query(q)) for q in queries]
