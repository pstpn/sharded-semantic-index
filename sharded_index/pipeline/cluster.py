"""Cluster the term graph with one of the configured methods."""

from __future__ import annotations

import logging

import pandas as pd

from sharded_index.config import Config
from sharded_index.graph.clustering import cluster_graph
from sharded_index.paths import Paths
from sharded_index.pipeline.common import method_argument, setup

logger = logging.getLogger(__name__)


def run(config: Config, paths: Paths, method: str) -> None:
    edges = pd.read_parquet(paths.graph)
    clustering = cluster_graph(edges, config.clustering.methods[method], config.clustering.seed)
    target = paths.clustering(method)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"term": list(clustering), "cluster": list(clustering.values())}).to_parquet(
        target, index=False
    )
    logger.info(
        "%s: %d terms in %d clusters", method, len(clustering), len(set(clustering.values()))
    )


if __name__ == "__main__":
    config, paths = setup()
    run(config, paths, method_argument(config))
