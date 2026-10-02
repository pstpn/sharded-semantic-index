"""Describe the collection, the graph and the clusterings.

Writes ``corpus.json``, ``clusterings.csv``, ``clusters.csv`` and ``frame_words.csv``.
"""

from __future__ import annotations

import json

import pandas as pd

from sharded_index.config import Config
from sharded_index.data.corpus import Corpus
from sharded_index.data.msmarco import split_queries
from sharded_index.paths import Paths
from sharded_index.pipeline.common import read_clustering, setup
from sharded_index.reporting.describe import (
    clusterings_table,
    clusters_table,
    collection_description,
    frame_word_candidates,
    graph_description,
)
from sharded_index.reporting.tables import write_table


def run(config: Config, paths: Paths) -> None:
    settings = config.describe
    pairs = pd.read_parquet(paths.pairs, columns=["query", "doc_id", "is_selected"])
    corpus = Corpus.load(paths.corpus)
    edges = pd.read_parquet(paths.graph)
    train_queries, holdout_queries = split_queries(pairs, config.queries.train_ratio)

    description = collection_description(pairs, corpus, train_queries, holdout_queries)
    description["graph"] = graph_description(edges, corpus, len(config.graph.stop_words))
    paths.metrics.mkdir(parents=True, exist_ok=True)
    (paths.metrics / "corpus.json").write_text(
        json.dumps(description, indent=2, ensure_ascii=False) + "\n"
    )

    clusterings = {method: read_clustering(paths, method) for method in config.clustering.methods}
    write_table(
        clusterings_table(edges, clusterings, config.clustering),
        paths.metrics / "clusterings.csv",
    )
    clusters = pd.concat(
        [
            clusters_table(edges, clustering, settings.top_clusters, settings.top_terms).assign(
                method=method
            )
            for method, clustering in clusterings.items()
        ],
        ignore_index=True,
    )
    write_table(
        clusters[["method", *clusters.columns.drop("method")]], paths.metrics / "clusters.csv"
    )
    write_table(
        frame_word_candidates(
            train_queries,
            corpus,
            config.tokenizer(),
            config.graph.stop_words,
            settings.frame_word_candidates,
        ),
        paths.metrics / "frame_words.csv",
    )


if __name__ == "__main__":
    run(*setup())
