"""Measure every strategy of one method on every query sample.

Writes the per-query measurements and the query-independent partition
statistics; the ``report`` stage aggregates them over all methods.
"""

from __future__ import annotations

import logging

import pandas as pd

from sharded_index.config import Config, Strategy
from sharded_index.data.corpus import Corpus
from sharded_index.evaluation.measure import measure_queries, partition_stats
from sharded_index.partition.strategies import hash_space_offset
from sharded_index.paths import Paths
from sharded_index.pipeline.common import (
    method_argument,
    read_clustering,
    read_partitions,
    read_rankings,
    setup,
)
from sharded_index.routing import Router, cover_weights
from sharded_index.storage import load_incidence

logger = logging.getLogger(__name__)


def run(config: Config, paths: Paths, method: str) -> None:
    corpus = Corpus.load(paths.corpus)
    clustering = read_clustering(paths, method)
    rankings = read_rankings(config, paths)
    weights = cover_weights(config.routing.cover_weight, corpus, pd.read_parquet(paths.graph))
    tokenizer = config.tokenizer()

    query_frames, partition_rows = [], []
    for name, partition in read_partitions(config, paths, method).items():
        strategy = Strategy.parse(name)
        offset = hash_space_offset(strategy, clustering, partition)
        membership = load_incidence(paths.assignments(method) / f"{name}.npz")
        router = Router(partition, weights, tokenizer)
        partition_rows.append(
            {"strategy": name, **partition_stats(partition, membership, corpus, strategy, offset)}
        )
        for sample, ranking in rankings.items():
            frame = measure_queries(
                router,
                membership,
                ranking,
                config.evaluation.budgets,
                config.evaluation.top_k,
                offset,
            )
            frame.insert(0, "sample", sample)
            frame.insert(0, "strategy", name)
            query_frames.append(frame)
        logger.info("%s/%s measured", method, name)

    target = paths.evaluation(method)
    target.mkdir(parents=True, exist_ok=True)
    pd.concat(query_frames, ignore_index=True).to_parquet(target / "queries.parquet", index=False)
    pd.DataFrame(partition_rows).to_parquet(target / "partitions.parquet", index=False)


if __name__ == "__main__":
    config, paths = setup()
    run(config, paths, method_argument(config))
