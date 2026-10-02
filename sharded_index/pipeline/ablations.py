"""Ablations of one method: variants of the configuration with one setting changed.

Every variant is rebuilt from the graph up, for every strategy, and measured
on one query sample.  A variant that only changes another method is skipped.
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


def _affects(config: Config, varied: Config, method: str) -> bool:
    """Whether a variant changes anything the results of the method depend on."""
    return (
        varied.graph != config.graph
        or varied.partition != config.partition
        or varied.routing != config.routing
        or varied.clustering.seed != config.clustering.seed
        or varied.clustering.methods[method] != config.clustering.methods[method]
    )


def run(config: Config, paths: Paths, method_name: str) -> None:
    settings = config.ablations
    pairs = pd.read_parquet(paths.pairs, columns=["query"])
    train_queries, _ = split_queries(pairs, config.queries.train_ratio)
    runner = VariantRunner(
        Corpus.load(paths.corpus), train_queries, read_rankings(config, paths)[settings.sample]
    )

    rows: list[dict[str, Any]] = []
    for name, overrides in settings.variants.items():
        varied = config.with_overrides(overrides)
        if overrides and not _affects(config, varied, method_name):
            continue
        results = runner.evaluate(
            varied, varied.clustering.methods[method_name], varied.partition.strategies
        )
        rows.extend({"variant": name, **row} for row in results)
        logger.info("%s: %s done", method_name, name)

    target = paths.ablations(method_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)


if __name__ == "__main__":
    config, paths = setup()
    run(config, paths, method_argument(config))
