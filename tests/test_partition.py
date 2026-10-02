from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sharded_index.config import ClusteringMethod, PartitionConfig, Strategy
from sharded_index.data.corpus import Corpus
from sharded_index.evaluation.membership import assign_documents, shard_volumes
from sharded_index.graph.clustering import cluster_graph
from sharded_index.partition.affinity import with_affinity_fallback
from sharded_index.partition.balance import balance
from sharded_index.partition.hashing import (
    hash_partition,
    hash_shard,
    id_space,
    with_hash_fallback,
)
from sharded_index.partition.model import TermPartition
from sharded_index.partition.replication import replicate
from sharded_index.partition.strategies import build_strategies, hash_space_offset
from synthetic import SEED, STRATEGIES

SPLITTERS = [
    ClusteringMethod(algorithm="leiden", resolution=1.0, iterations="convergence"),
    ClusteringMethod(algorithm="cpm", resolution=0.02, iterations=2),
    ClusteringMethod(algorithm="infomap"),
    ClusteringMethod(algorithm="metis", n_parts=4),
]


def test_partition_roundtrip_keeps_replica_order(tmp_path: Path) -> None:
    partition = TermPartition({"b": (3, 1), "a": (2,), "c": (0, 5, 4)})
    partition.save(tmp_path / "p.parquet")
    assert TermPartition.load(tmp_path / "p.parquet").term_shards == partition.term_shards


def test_partition_properties() -> None:
    partition = TermPartition({"a": (2,), "b": (3, 1)})
    assert partition.primary == {"a": 2, "b": 3}
    assert partition.n_shards == 4
    assert partition.replication_factor == 1.5
    assert partition.shards_of(["a", "b", "unknown"]) == {1, 2, 3}
    assert partition.incidence(["a", "b", "z"]).toarray().tolist() == [
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
    ]


def test_hash_shard_is_stable() -> None:
    assert hash_shard("apple", 7) == hash_shard("apple", 7)
    assert set(hash_partition(["b", "a"], 5).values()) <= set(range(5))
    assert list(hash_partition(["b", "a"], 5)) == ["a", "b"]


def test_hash_fallback_uses_a_separate_id_space(clustering: dict[str, int], corpus: Corpus) -> None:
    n = id_space(clustering)
    base = with_hash_fallback(clustering, corpus.vocabulary)
    assert list(base) == corpus.vocabulary
    for term, shard in base.items():
        if term in clustering:
            assert shard == clustering[term]
        else:
            assert n <= shard < 2 * n


def test_affinity_fallback_puts_tail_terms_into_clusters(
    clustering: dict[str, int], corpus: Corpus
) -> None:
    aff = with_affinity_fallback(clustering, corpus, "llr")
    assert list(aff) == corpus.vocabulary
    assert all(aff[term] == cluster for term, cluster in clustering.items() if term in aff)
    assert aff["windows10"] < id_space(clustering)
    assert aff == with_affinity_fallback(clustering, corpus, "llr")


def test_terms_without_graph_neighbours_are_hashed(clustering: dict[str, int]) -> None:
    lonely = Corpus.from_token_sets(["d"], [{"zzzalone"}])
    aff = with_affinity_fallback(clustering, lonely, "llr")
    assert aff["zzzalone"] >= id_space(clustering)


def test_vote_follows_significance_not_popularity() -> None:
    clustering = {"common": 0, "usual": 0, "topic": 1}
    token_sets = [{"common", "usual"}] * 20 + [{"common", "usual", "topic", "rare"}] * 3
    corpus = Corpus.from_token_sets([f"d{i}" for i in range(len(token_sets))], token_sets)
    assert with_affinity_fallback(clustering, corpus, "raw")["rare"] == 0
    assert with_affinity_fallback(clustering, corpus, "llr")["rare"] == 1


@pytest.mark.parametrize("method", SPLITTERS, ids=lambda method: method.algorithm)
def test_volume_budget_holds_for_every_shard(
    edges: pd.DataFrame,
    corpus: Corpus,
    partition_config: PartitionConfig,
    method: ClusteringMethod,
) -> None:
    clustering = cluster_graph(edges, method, SEED)
    aff = with_affinity_fallback(clustering, corpus, "llr")
    settings = partition_config.balance
    balanced = balance(aff, corpus, edges, method, SEED, settings, id_space(clustering))

    assert list(balanced) == corpus.vocabulary
    assert set(balanced.values()) == set(range(len(set(balanced.values()))))
    assert balanced == balance(aff, corpus, edges, method, SEED, settings, id_space(clustering))

    partition = TermPartition.from_primary(balanced)
    volumes = shard_volumes(assign_documents(partition, corpus))
    terms_in_shard = np.bincount(list(balanced.values()))
    budget = settings.volume_cap * corpus.n_docs
    assert all(
        volume <= budget or terms_in_shard[shard] == 1 for shard, volume in enumerate(volumes)
    )


def test_balance_requires_the_whole_vocabulary(
    clustering: dict[str, int],
    edges: pd.DataFrame,
    corpus: Corpus,
    leiden: ClusteringMethod,
    partition_config: PartitionConfig,
) -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        balance(clustering, corpus, edges, leiden, SEED, partition_config.balance, 0)


def test_replicas_keep_the_primary_shard_and_stay_semantic(
    clustering: dict[str, int], edges: pd.DataFrame, corpus: Corpus
) -> None:
    base = with_hash_fallback(clustering, corpus.vocabulary)
    replicated = replicate(base, edges, 2)
    assert set(replicated.term_shards) == set(base)
    for term, shards in replicated.term_shards.items():
        assert shards[0] == base[term]
        assert len(set(shards)) == len(shards) <= 2
        if term in clustering:
            assert all(shard < id_space(clustering) for shard in shards)
        else:
            assert len(shards) == 1
    assert 1.0 <= replicated.replication_factor <= 2.0
    assert replicate(base, edges, 1).term_shards == TermPartition.from_primary(base).term_shards


def test_affinity_quantile_only_removes_replicas(
    clustering: dict[str, int], edges: pd.DataFrame, corpus: Corpus
) -> None:
    base = with_hash_fallback(clustering, corpus.vocabulary)
    everything = replicate(base, edges, 2, affinity_quantile=0.0)
    strongest = replicate(base, edges, 2, affinity_quantile=1.0)
    assert everything.term_shards == replicate(base, edges, 2).term_shards
    assert strongest.replication_factor <= everything.replication_factor


def test_every_configured_strategy_is_built(
    strategies: dict[str, TermPartition], corpus: Corpus
) -> None:
    assert tuple(strategies) == STRATEGIES
    for partition in strategies.values():
        assert list(partition.term_shards) == corpus.vocabulary


def test_hash_baselines_mirror_the_shard_count_of_their_family(
    strategies: dict[str, TermPartition], corpus: Corpus
) -> None:
    for family in ("base", "aff", "bal"):
        n_shards = len(set(strategies[family].primary.values()))
        expected = {term: (hash_shard(term, n_shards),) for term in corpus.vocabulary}
        assert strategies[f"hash_{family}"].term_shards == expected


def test_strategies_can_be_built_selectively(
    clustering: dict[str, int],
    edges: pd.DataFrame,
    corpus: Corpus,
    leiden: ClusteringMethod,
    partition_config: PartitionConfig,
    strategies: dict[str, TermPartition],
) -> None:
    only = build_strategies(
        clustering, edges, corpus, leiden, SEED, partition_config, ["hash_bal", "aff_r2"]
    )
    assert list(only) == ["hash_bal", "aff_r2"]
    assert only["aff_r2"].term_shards == strategies["aff_r2"].term_shards
    with pytest.raises(ValueError, match="strategy"):
        build_strategies(clustering, edges, corpus, leiden, SEED, partition_config, ["bal_r2"])


def test_hash_space_offsets(
    strategies: dict[str, TermPartition], clustering: dict[str, int]
) -> None:
    def offset(name: str) -> int | None:
        return hash_space_offset(Strategy.parse(name), clustering, strategies[name])

    assert offset("base") == offset("aff_r3") == id_space(clustering)
    assert offset("bal") == strategies["bal"].n_shards
    assert offset("hash_bal") is None
