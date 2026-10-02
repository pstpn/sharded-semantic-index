from __future__ import annotations

import pandas as pd
import pytest

from sharded_index.config import ClusteringMethod, GraphConfig
from sharded_index.graph.clustering import cluster_graph, modularity, refinements
from sharded_index.graph.cooccurrence import build_graph, graph_terms, node_strength
from synthetic import SEED, STOP_WORDS

LOOSE = GraphConfig(
    min_df=1,
    max_df_ratio=1.0,
    min_pair_count=1,
    min_npmi=-1.0,
    edge_weight="npmi",
    stop_words=(),
)
METHODS = [
    ClusteringMethod(algorithm="leiden", resolution=1.0, iterations="convergence"),
    ClusteringMethod(algorithm="cpm", resolution=0.05, iterations=2),
    ClusteringMethod(algorithm="infomap"),
    ClusteringMethod(algorithm="metis", n_parts=4),
]


def test_repeated_and_rephrased_queries_count_once() -> None:
    unique = build_graph(["apple juice", "dog cat", "apple dog"], LOOSE, STOP_WORDS)
    repeated = build_graph(["apple juice", "dog cat"] * 3 + ["apple dog"], LOOSE, STOP_WORDS)
    rephrased = build_graph(
        ["apple juice", "the apple juice", "dog cat", "cat dog", "apple dog"], LOOSE, STOP_WORDS
    )
    assert repeated.equals(unique)
    assert rephrased.equals(unique)


def test_stop_words_and_digit_tokens_stay_out_of_the_graph(edges: pd.DataFrame) -> None:
    terms = set(graph_terms(edges))
    assert terms
    assert "the" not in terms
    assert "windows10" not in terms


def test_edge_weights_share_one_topology(texts: list[str], graph_config: GraphConfig) -> None:
    def topology(weight: str) -> set[tuple[str, str, int]]:
        graph = build_graph(
            texts, graph_config.model_copy(update={"edge_weight": weight}), STOP_WORDS
        )
        return set(zip(graph["src"], graph["dst"], graph["count"], strict=True))

    assert topology("npmi") == topology("llr") == topology("chi2")


def test_edges_are_sorted_by_weight(edges: pd.DataFrame) -> None:
    assert edges["weight"].is_monotonic_decreasing


def test_node_strength_sums_incident_weights() -> None:
    edges = pd.DataFrame(
        {"src": ["a", "a"], "dst": ["b", "c"], "count": [2, 2], "weight": [0.5, 0.25]}
    )
    assert node_strength(edges) == {"a": 0.75, "b": 0.5, "c": 0.25}


@pytest.mark.parametrize("method", METHODS, ids=lambda method: method.algorithm)
def test_clustering_covers_the_graph_and_is_reproducible(
    edges: pd.DataFrame, method: ClusteringMethod
) -> None:
    clustering = cluster_graph(edges, method, SEED)
    assert set(clustering) == set(graph_terms(edges))
    assert len(set(clustering.values())) > 1
    assert clustering == cluster_graph(edges, method, SEED)


def test_topics_are_recovered(edges: pd.DataFrame, clustering: dict[str, int]) -> None:
    assert clustering["apple"] == clustering["juice"]
    assert clustering["apple"] != clustering["dog"]
    assert 0 < modularity(edges, clustering) <= 1


def test_refinements_get_progressively_finer() -> None:
    leiden = ClusteringMethod(algorithm="leiden", resolution=1.5, iterations=2)
    assert [m.resolution for m in refinements(leiden, 3.2, 100, 3)] == [1.5, 3.0, 6.0]
    metis = ClusteringMethod(algorithm="metis", n_parts=50)
    assert [m.n_parts for m in refinements(metis, 3.2, 100, 5)] == [4, 8, 16, 32, 64]
    assert [m.n_parts for m in refinements(metis, 3.2, 10, 3)] == [4, 8, 10]
    infomap = ClusteringMethod(algorithm="infomap")
    assert list(refinements(infomap, 3.2, 100, 5)) == [infomap]
