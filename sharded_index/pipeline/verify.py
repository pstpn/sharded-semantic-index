"""Verify that the emulated evaluation of one method is exact.

For every strategy: completeness over all documents and all evaluated
queries, and physical equivalence on real shard indices for a few queries.
The stage fails if any check does not pass.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys

import pandas as pd

from sharded_index.config import Config
from sharded_index.data.corpus import Corpus
from sharded_index.data.queries import sample_queries
from sharded_index.evaluation.verification import (
    PhysicalCheck,
    check_completeness,
    check_reference,
    reference_results,
)
from sharded_index.index.ranking import ReferenceIndex
from sharded_index.paths import Paths
from sharded_index.pipeline.common import (
    method_argument,
    read_documents,
    read_partitions,
    read_rankings,
    setup,
)
from sharded_index.routing import Router, cover_weights
from sharded_index.storage import load_incidence

logger = logging.getLogger(__name__)


def run(config: Config, paths: Paths, method: str) -> bool:
    settings = config.verification
    corpus = Corpus.load(paths.corpus)
    documents = read_documents(paths)
    tokenizer = config.tokenizer()
    token_sets = [set(tokenizer(text)) for text in documents.values()]
    weights = cover_weights(config.routing.cover_weight, corpus, pd.read_parquet(paths.graph))

    rankings = read_rankings(config, paths)
    evaluated = {
        sample: [
            query for query, kept in zip(ranking.queries, ranking.evaluated, strict=True) if kept
        ]
        for sample, ranking in rankings.items()
    }
    checked = rankings[settings.sample]
    queries = sample_queries(evaluated[settings.sample], settings.queries, settings.seed)
    reference = ReferenceIndex(paths.index, corpus.doc_ids, config.bm25)
    references = reference_results(reference, queries, corpus.doc_ids)
    reference_checks = check_reference(
        references,
        dict(zip(checked.queries, checked.arrays, strict=True)),
        settings.score_tolerance,
    )
    physical_check = PhysicalCheck(
        reference,
        references,
        documents,
        corpus.doc_ids,
        config.evaluation.budgets,
        config.bm25,
        settings.score_tolerance,
        settings.workers,
    )

    strategies = {}
    for name, partition in read_partitions(config, paths, method).items():
        membership = load_incidence(paths.assignments(method) / f"{name}.npz")
        router = Router(partition, weights, tokenizer)
        completeness = check_completeness(partition, membership, token_sets, router, evaluated)
        physical = physical_check.run(membership, router, paths.verification_shards(method) / name)
        passed = (
            completeness["assignment_exact"]
            and completeness["incomplete_covers"] == 0
            and physical["max_score_difference"] <= settings.score_tolerance
            and physical["result_mismatches"] == 0
        )
        strategies[name] = {"completeness": completeness, "physical": physical, "passed": passed}
        logger.info("%s/%s: %s", method, name, "passed" if passed else "FAILED")
    reference.close()
    shutil.rmtree(paths.verification_shards(method), ignore_errors=True)

    passed = all(checks["passed"] for checks in strategies.values()) and not any(
        reference_checks.values()
    )
    result = {
        "method": method,
        "document_term_pairs": int(corpus.doc_terms.nnz),
        "reference": {"queries": len(queries), **reference_checks},
        "strategies": strategies,
        "passed": passed,
    }
    target = paths.verification(method)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    return passed


if __name__ == "__main__":
    config, paths = setup()
    if not run(config, paths, method_argument(config)):
        sys.exit("verification failed: see the report in data/interim/verification")
