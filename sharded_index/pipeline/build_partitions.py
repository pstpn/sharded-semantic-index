"""Build the term partition of every strategy from the clustering of one method."""

from __future__ import annotations

import logging

import pandas as pd

from sharded_index.config import Config
from sharded_index.data.corpus import Corpus
from sharded_index.partition.strategies import build_strategies
from sharded_index.paths import Paths
from sharded_index.pipeline.common import method_argument, read_clustering, setup

logger = logging.getLogger(__name__)


def run(config: Config, paths: Paths, method: str) -> None:
    partitions = build_strategies(
        read_clustering(paths, method),
        pd.read_parquet(paths.graph),
        Corpus.load(paths.corpus),
        config.clustering.methods[method],
        config.clustering.seed,
        config.partition,
    )
    for name, partition in partitions.items():
        partition.save(paths.partitions(method) / f"{name}.parquet")
        logger.info(
            "%s/%s: %d shards, replication factor %.3f",
            method,
            name,
            len({shard for shards in partition.term_shards.values() for shard in shards}),
            partition.replication_factor,
        )


if __name__ == "__main__":
    config, paths = setup()
    run(config, paths, method_argument(config))
