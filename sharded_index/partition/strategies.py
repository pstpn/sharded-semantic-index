"""The benchmarked strategies, all built from one clustering of the term graph.

Families (how a clustering becomes a partition of the corpus vocabulary):

- ``base`` — graph terms keep their cluster, the rest are hashed into a separate id space;
- ``aff`` — out-of-graph terms join a cluster by the document vote;
- ``bal`` — ``aff`` re-partitioned under the shard volume budget.

``base_r<k>`` and ``aff_r<k>`` replicate graph terms into their top-``k``
clusters; ``hash_<family>`` is the random baseline with as many shards as
the family has.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from sharded_index.config import ClusteringMethod, PartitionConfig, Strategy
from sharded_index.data.corpus import Corpus
from sharded_index.partition.affinity import with_affinity_fallback
from sharded_index.partition.balance import balance
from sharded_index.partition.hashing import hash_partition, id_space, with_hash_fallback
from sharded_index.partition.model import TermPartition
from sharded_index.partition.replication import replicate


def build_strategies(
    clustering: Mapping[str, int],
    edges: pd.DataFrame,
    corpus: Corpus,
    method: ClusteringMethod,
    seed: int | None,
    config: PartitionConfig,
    names: Iterable[str] | None = None,
) -> dict[str, TermPartition]:
    """Build the named strategies (all of ``config.strategies`` by default).

    ``method`` and ``seed`` are the settings that produced ``clustering``:
    the volume budget splits oversized clusters with the same algorithm.
    """
    families: dict[str, dict[str, int]] = {}

    def family(kind: str) -> dict[str, int]:
        if kind not in families:
            if kind == "base":
                families[kind] = with_hash_fallback(clustering, corpus.vocabulary)
            elif kind == "aff":
                families[kind] = with_affinity_fallback(clustering, corpus, config.affinity_vote)
            else:
                families[kind] = balance(
                    family("aff"),
                    corpus,
                    edges,
                    method,
                    seed,
                    config.balance,
                    hash_offset=id_space(clustering),
                )
        return families[kind]

    built: dict[str, TermPartition] = {}
    for name in config.strategies if names is None else names:
        strategy = Strategy.parse(name)
        primary = family(strategy.family)
        if strategy.is_hash:
            n_shards = len(set(primary.values()))
            built[name] = TermPartition.from_primary(hash_partition(corpus.vocabulary, n_shards))
        elif strategy.replicas > 1:
            built[name] = replicate(
                primary, edges, strategy.replicas, config.replica_affinity_quantile
            )
        else:
            built[name] = TermPartition.from_primary(primary)
    return built


def hash_space_offset(
    strategy: Strategy,
    clustering: Mapping[str, int],
    partition: TermPartition,
) -> int | None:
    """First shard id of the hash-fallback space of a strategy.

    ``base`` and ``aff`` keep hashed terms in ``[n, 2n)`` after the ``n``
    cluster ids.  ``bal`` packs every term into budgeted shards, so its hash
    space is empty.  For the hash baselines the notion does not apply (``None``).
    """
    if strategy.is_hash:
        return None
    if strategy.family == "bal":
        return partition.n_shards
    return id_space(clustering)
