"""Checks that make the emulated evaluation trustworthy.

Completeness — the full cover of a query returns everything the unsharded
index returns: the stored document assignment follows the rule "a document
lives in every shard of each of its terms", and the cover of every query
holds a shard of each of its terms.

Physical equivalence — real shard indices searched with collection-wide BM25
statistics return what the emulation says: every document's shard score
equals its score in the unsharded index, and for every budget the emulated
result equals the merged result of the probed shards.
"""

from __future__ import annotations

import shutil
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from sharded_index.config import Bm25Config
from sharded_index.evaluation.measure import budget_labels
from sharded_index.evaluation.membership import assign_by_lookup, same_membership
from sharded_index.index.bm25 import GlobalBM25
from sharded_index.index.build import build_shard_indices
from sharded_index.index.ranking import ReferenceIndex
from sharded_index.index.search import ShardSearcher
from sharded_index.partition.model import TermPartition
from sharded_index.routing import Router


@dataclass(frozen=True)
class ReferenceResult:
    """Everything the unsharded index returns for one query."""

    parsed: Any
    ranking: np.ndarray
    scores: np.ndarray
    score_of: dict[str, float]


def cover_is_complete(router: Router, query: str, cover: Sequence[int]) -> bool:
    """Whether the cover holds a shard of every query term the partition knows."""
    probed = set(cover)
    term_shards = router.partition.term_shards
    return all(
        probed.intersection(term_shards[term])
        for term in set(router.tokenizer(query))
        if term in term_shards
    )


def check_completeness(
    partition: TermPartition,
    membership: sp.csr_matrix,
    token_sets: Sequence[Collection[str]],
    router: Router,
    samples: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Exact completeness of one strategy over all documents and all evaluated queries."""
    incomplete = {
        sample: sum(not cover_is_complete(router, query, router.cover(query)) for query in queries)
        for sample, queries in samples.items()
    }
    return {
        "assignment_exact": same_membership(membership, assign_by_lookup(partition, token_sets)),
        "incomplete_covers": sum(incomplete.values()),
        "incomplete_covers_by_sample": incomplete,
        "queries_checked": sum(len(queries) for queries in samples.values()),
    }


def reference_results(
    reference: ReferenceIndex,
    queries: Sequence[str],
    doc_ids: Sequence[str],
) -> dict[str, ReferenceResult]:
    """Complete rankings with scores of the unsharded index for the given queries."""
    results = {}
    for query in queries:
        ranking, scores = reference.search(query, limit=None)
        score_of = {doc_ids[doc]: float(score) for doc, score in zip(ranking, scores, strict=True)}
        results[query] = ReferenceResult(reference.parse(query), ranking, scores, score_of)
    return results


def check_reference(
    references: Mapping[str, ReferenceResult],
    saved_rankings: Mapping[str, np.ndarray],
    tolerance: float,
) -> dict[str, int]:
    """The rankings the evaluation used are the head of the BM25 order of the unsharded index."""
    order_violations = prefix_mismatches = 0
    for query, result in references.items():
        order_violations += bool(np.any(np.diff(result.scores) > tolerance))
        saved = saved_rankings[query]
        prefix_mismatches += saved.tolist() != result.ranking[: len(saved)].tolist()
    return {
        "ranking_order_violations": order_violations,
        "ranking_prefix_mismatches": prefix_mismatches,
    }


def _same_scores(emulated: np.ndarray, physical: np.ndarray, tolerance: float) -> bool:
    """Equal score sequences: the same result up to swaps of equally scored documents."""
    return len(emulated) == len(physical) and bool(np.all(np.abs(emulated - physical) <= tolerance))


@dataclass(frozen=True)
class PhysicalCheck:
    """Physical equivalence check; holds what every strategy's check shares."""

    reference: ReferenceIndex
    references: Mapping[str, ReferenceResult]
    documents: Mapping[str, str]
    doc_ids: Sequence[str]
    budgets: Sequence[int]
    bm25: Bm25Config
    tolerance: float
    workers: int | None = None

    def run(self, membership: sp.csr_matrix, router: Router, shards_root: Path) -> dict[str, Any]:
        """Build the real shard indices of the queries' covers and compare them with the emulation.

        The indices are built under ``shards_root`` and removed afterwards.
        """
        covers = {query: router.cover(query) for query in self.references}
        needed = sorted({shard for cover in covers.values() for shard in cover})
        by_shard = membership.tocsc()
        shard_documents = {
            shard: {
                self.doc_ids[doc]: self.documents[self.doc_ids[doc]]
                for doc in by_shard.indices[by_shard.indptr[shard] : by_shard.indptr[shard + 1]]
            }
            for shard in needed
        }
        build_shard_indices(shard_documents, shards_root, router.tokenizer.stop_words, self.workers)
        weighting = GlobalBM25(self.reference.searcher, b=self.bm25.b, k1=self.bm25.k1)
        searcher = ShardSearcher(shards_root, weighting)

        labels = budget_labels(self.budgets)
        max_difference, compared = 0.0, 0
        mismatches = dict.fromkeys(labels, 0)
        for query, result in self.references.items():
            cover = covers[query]
            shard_scores = {shard: searcher.scores(shard, result.parsed) for shard in cover}
            for scores in shard_scores.values():
                for doc_id, score in scores.items():
                    difference = abs(score - result.score_of.get(doc_id, float("inf")))
                    max_difference = max(max_difference, difference)
                    compared += 1

            for label, budget in zip(labels, [*self.budgets, len(cover)], strict=True):
                probed = cover[:budget]
                in_probed = (
                    membership[result.ranking][:, probed].toarray().any(axis=1)
                    if probed and len(result.ranking)
                    else np.zeros(len(result.ranking), dtype=bool)
                )
                merged: dict[str, float] = {}
                for shard in probed:
                    merged.update(shard_scores[shard])
                physical = np.sort(np.fromiter(merged.values(), dtype=float))[::-1]
                same = _same_scores(result.scores[in_probed], physical, self.tolerance)
                mismatches[label] += not same

        shutil.rmtree(shards_root)
        return {
            "shards_built": len(needed),
            "documents_indexed": sum(len(docs) for docs in shard_documents.values()),
            "scores_compared": compared,
            "max_score_difference": max_difference,
            "result_mismatches": sum(mismatches.values()),
            "result_mismatches_by_budget": mismatches,
        }
