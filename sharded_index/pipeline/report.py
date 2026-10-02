"""Aggregate the measurements of all methods into the result tables of ``metrics/``."""

from __future__ import annotations

import json
import logging

import pandas as pd

from sharded_index.config import Config
from sharded_index.evaluation.slices import connectivity_slices, term_count_slices
from sharded_index.evaluation.statistics import bootstrap_weights
from sharded_index.paths import Paths
from sharded_index.pipeline.common import read_samples, setup
from sharded_index.reporting.tables import (
    comparison_pairs,
    comparisons_table,
    group_queries,
    retrieval_table,
    routing_table,
    slices_table,
    variants_table,
    verification_table,
    write_table,
)

logger = logging.getLogger(__name__)


def run(config: Config, paths: Paths) -> None:
    methods = list(config.clustering.methods)
    strategies = list(config.partition.strategies)
    samples = config.queries.sample_names()
    evaluation = config.evaluation
    budgets, confidence = evaluation.budgets, evaluation.bootstrap.confidence

    partitions = {
        method: pd.read_parquet(paths.evaluation(method) / "partitions.parquet")
        for method in methods
    }
    write_table(
        variants_table(partitions, ["method", "strategy"]), paths.metrics / "partitions.csv"
    )

    groups = group_queries(
        pd.concat(
            [
                pd.read_parquet(paths.evaluation(method) / "queries.parquet").assign(method=method)
                for method in methods
            ],
            ignore_index=True,
        )
    )
    weights = {
        sample: bootstrap_weights(
            len(groups[methods[0], strategies[0], sample]),
            evaluation.bootstrap.samples,
            evaluation.bootstrap.seed,
        )
        for sample in samples
    }
    keys = [
        (method, strategy, sample)
        for method in methods
        for strategy in strategies
        for sample in samples
    ]
    write_table(routing_table(groups, keys, weights, confidence), paths.metrics / "routing.csv")
    write_table(
        retrieval_table(groups, keys, budgets, weights, confidence), paths.metrics / "retrieval.csv"
    )
    pairs = comparison_pairs(methods, strategies, evaluation.comparisons)
    write_table(
        comparisons_table(groups, pairs, samples, budgets, weights, confidence),
        paths.metrics / "comparisons.csv",
    )
    logger.info("routing, retrieval and comparisons written")

    sliced = evaluation.slices.sample
    queries = groups[methods[0], strategies[0], sliced]
    evaluated = queries["evaluated"].tolist()
    slices = pd.concat(
        [
            connectivity_slices(
                read_samples(config, paths)[sliced],
                evaluated,
                pd.read_parquet(paths.graph),
                config.tokenizer(),
                evaluation.slices.connectivity_quantiles,
            ),
            term_count_slices(queries["terms"].tolist(), evaluated, evaluation.slices.max_terms),
        ],
        ignore_index=True,
    )
    write_table(
        slices_table(
            groups,
            [(method, strategy) for method in methods for strategy in strategies],
            sliced,
            slices,
            budgets,
        ),
        paths.metrics / "slices.csv",
    )

    sensitivity = {method: pd.read_parquet(paths.sensitivity(method)) for method in methods}
    write_table(
        variants_table(sensitivity, ["sweep", "method", "value", "strategy"]),
        paths.metrics / "sensitivity.csv",
    )
    ablations = {method: pd.read_parquet(paths.ablations(method)) for method in methods}
    write_table(
        variants_table(ablations, ["variant", "method", "strategy"]),
        paths.metrics / "ablations.csv",
    )

    verification = {
        method: json.loads(paths.verification(method).read_text()) for method in methods
    }
    write_table(verification_table(verification), paths.metrics / "verification.csv")
    logger.info("tables written to %s", paths.metrics)


if __name__ == "__main__":
    run(*setup())
