"""Draw the evaluation query samples: train, holdout, out-of-distribution and synthetic."""

from __future__ import annotations

import logging

import pandas as pd

from sharded_index.config import Config
from sharded_index.data.msmarco import split_queries
from sharded_index.data.queries import build_samples
from sharded_index.paths import Paths
from sharded_index.pipeline.common import setup

logger = logging.getLogger(__name__)


def run(config: Config, paths: Paths) -> None:
    pairs = pd.read_parquet(paths.pairs, columns=["query"])
    train_queries, holdout_queries = split_queries(pairs, config.queries.train_ratio)
    samples = build_samples(
        train_queries, holdout_queries, pd.read_parquet(paths.graph), config.queries
    )
    paths.queries.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(paths.queries, index=False)
    for sample, size in samples.groupby("sample", sort=False).size().items():
        logger.info("%s: %d queries", sample, size)


if __name__ == "__main__":
    run(*setup())
