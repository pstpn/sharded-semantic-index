"""Build the term co-occurrence graph from the training queries."""

from __future__ import annotations

import logging

import pandas as pd

from sharded_index.config import Config
from sharded_index.data.msmarco import split_queries
from sharded_index.graph.cooccurrence import build_graph, graph_terms
from sharded_index.paths import Paths
from sharded_index.pipeline.common import setup

logger = logging.getLogger(__name__)


def run(config: Config, paths: Paths) -> None:
    pairs = pd.read_parquet(paths.pairs, columns=["query"])
    train_queries, holdout_queries = split_queries(pairs, config.queries.train_ratio)
    edges = build_graph(train_queries, config.graph, config.graph_stop_words())
    paths.graph.parent.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(paths.graph, index=False)
    logger.info(
        "%d training queries (%d held out): %d terms, %d edges",
        len(train_queries),
        len(holdout_queries),
        len(graph_terms(edges)),
        len(edges),
    )


if __name__ == "__main__":
    run(*setup())
