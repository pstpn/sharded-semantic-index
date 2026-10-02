"""In-memory evaluation of configuration variants (sensitivity sweeps and ablations).

A variant goes through the same steps as the main pipeline — graph,
clustering, strategies, document assignment, measurement — without writing
the intermediate artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sharded_index.config import ClusteringMethod, Config, Strategy
from sharded_index.data.corpus import Corpus
from sharded_index.evaluation.measure import budget_labels, measure_queries, partition_stats
from sharded_index.evaluation.membership import assign_documents
from sharded_index.evaluation.statistics import retrieval_summary, routing_summary
from sharded_index.graph.clustering import cluster_graph
from sharded_index.graph.cooccurrence import build_graph
from sharded_index.index.ranking import Rankings
from sharded_index.partition.strategies import build_strategies, hash_space_offset
from sharded_index.routing import Router, cover_weights
from sharded_index.text import Tokenizer


@dataclass
class VariantRunner:
    """Evaluates variants on one query sample; graphs and clusterings are reused across calls."""

    corpus: Corpus
    train_queries: list[str]
    rankings: Rankings
    _graphs: dict[Any, pd.DataFrame] = field(default_factory=dict, repr=False)
    _clusterings: dict[Any, dict[str, int]] = field(default_factory=dict, repr=False)

    def _graph(self, config: Config, n_train: int | None) -> tuple[Any, pd.DataFrame]:
        key = (config.graph, config.graph_stop_words(), n_train)
        if key not in self._graphs:
            queries = self.train_queries if n_train is None else self.train_queries[:n_train]
            self._graphs[key] = build_graph(queries, config.graph, config.graph_stop_words())
        return key, self._graphs[key]

    def evaluate(
        self,
        config: Config,
        method: ClusteringMethod,
        strategies: tuple[str, ...],
        *,
        seed: int | None = None,
        n_train: int | None = None,
    ) -> list[dict[str, Any]]:
        """Summary rows, one per strategy, of a variant built with ``config`` and ``method``.

        ``seed`` overrides ``clustering.seed``; ``n_train`` builds the graph
        from the first ``n_train`` training queries only.
        """
        seed = config.clustering.seed if seed is None else seed
        graph_key, edges = self._graph(config, n_train)
        clustering_key = (graph_key, method, seed)
        if clustering_key not in self._clusterings:
            self._clusterings[clustering_key] = cluster_graph(edges, method, seed)
        clustering = self._clusterings[clustering_key]

        tokenizer = config.tokenizer()
        weights = cover_weights(config.routing.cover_weight, self.corpus, edges)
        evaluation = config.evaluation
        graph_summary = {
            "graph_terms": len(clustering),
            "clusters": len(set(clustering.values())),
            "queries_in_graph": self._queries_in_graph(tokenizer, set(clustering)),
        }

        partitions = build_strategies(
            clustering, edges, self.corpus, method, seed, config.partition, strategies
        )
        rows = []
        for name, partition in partitions.items():
            strategy = Strategy.parse(name)
            offset = hash_space_offset(strategy, clustering, partition)
            membership = assign_documents(partition, self.corpus)
            frame = measure_queries(
                Router(partition, weights, tokenizer),
                membership,
                self.rankings,
                evaluation.budgets,
                evaluation.top_k,
                offset,
            )
            stats = partition_stats(partition, membership, self.corpus, strategy, offset)
            rows.append(
                {
                    "strategy": name,
                    **graph_summary,
                    **variant_summary(stats, frame, evaluation.budgets),
                }
            )
        return rows

    def _queries_in_graph(self, tokenizer: Tokenizer, graph_terms: set[str]) -> float:
        """Share of the evaluated queries whose every term is a graph term."""
        evaluated = zip(self.rankings.queries, self.rankings.evaluated, strict=True)
        query_terms = [set(tokenizer(query)) for query, kept in evaluated if kept]
        covered = [bool(terms) and terms <= graph_terms for terms in query_terms]
        return float(np.mean(covered)) if covered else float("nan")


def variant_summary(
    stats: dict[str, float], frame: pd.DataFrame, budgets: tuple[int, ...]
) -> dict[str, float]:
    """The metrics reported for every variant: partition cost, cover and retrieval by budget."""
    routing = routing_summary(frame)
    retrieval = retrieval_summary(frame, budgets)
    labels = budget_labels(budgets)
    return {
        "shards": stats["shards"],
        "duplication": stats["duplication"],
        "largest_shard": stats["largest_shard"],
        "fanout_mean": routing["fanout_mean"],
        "fanout_p95": routing["fanout_p95"],
        "single_shard_share": routing["single_shard_share"],
        **{f"overlap_{label}": retrieval[label]["overlap"] for label in labels},
        **{f"volume_{label}": retrieval[label]["volume"] for label in labels},
    }
