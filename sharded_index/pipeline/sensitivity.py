"""Sensitivity of one method to its seed, its own parameter and the size of the query log.

Each sweep changes one thing, rebuilds every strategy and measures it on one
query sample.  The first row of every sweep is the main configuration.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from sharded_index.config import Config
from sharded_index.data.corpus import Corpus
from sharded_index.data.msmarco import split_queries
from sharded_index.evaluation.variants import VariantRunner
from sharded_index.paths import Paths
from sharded_index.pipeline.common import method_argument, read_rankings, setup

logger = logging.getLogger(__name__)


def run(config: Config, paths: Paths, method_name: str) -> None:
    settings = config.sensitivity
    method = config.clustering.methods[method_name]
    strategies = config.partition.strategies
    pairs = pd.read_parquet(paths.pairs, columns=["query"])
    train_queries, _ = split_queries(pairs, config.queries.train_ratio)
    runner = VariantRunner(
        Corpus.load(paths.corpus), train_queries, read_rankings(config, paths)[settings.sample]
    )
    rows: list[dict[str, Any]] = []

    def record(sweep: str, value: float, results: list[dict[str, Any]]) -> None:
        rows.extend({"sweep": sweep, "value": float(value), **row} for row in results)
        logger.info("%s: %s=%s done", method_name, sweep, value)

    main = runner.evaluate(config, method, strategies)

    record("seed", config.clustering.seed, main)
    for seed in settings.seeds:
        if seed != config.clustering.seed:
            record("seed", seed, runner.evaluate(config, method, strategies, seed=seed))

    for parameter, values in settings.parameters.get(method_name, {}).items():
        current = getattr(method, parameter)
        record(parameter, current, main)
        for value in values:
            if value != current:
                varied = method.with_parameter(parameter, value)
                record(parameter, value, runner.evaluate(config, varied, strategies))

    record("train_size", len(train_queries), main)
    for size in sorted(settings.train_sizes):
        if size < len(train_queries):
            record("train_size", size, runner.evaluate(config, method, strategies, n_train=size))

    target = paths.sensitivity(method_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)


if __name__ == "__main__":
    config, paths = setup()
    run(config, paths, method_argument(config))
