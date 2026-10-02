"""Term replication: a graph term is also placed into the clusters it is most tied to."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from sharded_index.partition.model import TermPartition


def cluster_affinity(edges: pd.DataFrame, term_to_shard: Mapping[str, int]) -> pd.DataFrame:
    """Total edge weight between every term and every cluster it has edges into.

    Returns ``[term, cluster, affinity]``.
    """
    if edges.empty:
        return pd.DataFrame(columns=["term", "cluster", "affinity"])
    directed = pd.concat(
        [
            pd.DataFrame(
                {
                    "term": edges["src"],
                    "cluster": edges["dst"].map(term_to_shard),
                    "affinity": edges["weight"],
                }
            ),
            pd.DataFrame(
                {
                    "term": edges["dst"],
                    "cluster": edges["src"].map(term_to_shard),
                    "affinity": edges["weight"],
                }
            ),
        ],
        ignore_index=True,
    )
    return directed.groupby(["term", "cluster"], as_index=False)["affinity"].sum()


def replicate(
    primary: Mapping[str, int],
    edges: pd.DataFrame,
    n_replicas: int,
    affinity_quantile: float | None = None,
) -> TermPartition:
    """Give every graph term up to ``n_replicas`` shards: its own and its top-affinity clusters.

    Documents replicate with their terms, so duplication grows, but
    co-occurring query terms more often share a shard.  Terms without edges
    into other clusters keep a single shard.  ``affinity_quantile`` keeps
    only the candidates at or above that quantile of all cross-cluster
    affinities.
    """
    if n_replicas < 1:
        msg = "n_replicas must be at least 1"
        raise ValueError(msg)

    ranked: dict[str, list[int]] = {}
    affinity = cluster_affinity(edges, primary)
    if not affinity.empty:
        other = affinity[affinity["cluster"] != affinity["term"].map(primary)]
        if affinity_quantile is not None and not other.empty:
            threshold = other["affinity"].quantile(affinity_quantile)
            other = other[other["affinity"] >= threshold]
        if not other.empty:
            other = other.sort_values(
                ["term", "affinity", "cluster"],
                ascending=[True, False, True],
                kind="stable",
            )
            ranked = other.groupby("term", sort=False)["cluster"].agg(list).to_dict()

    return TermPartition(
        {
            term: (shard, *(int(cluster) for cluster in ranked.get(term, [])[: n_replicas - 1]))
            for term, shard in primary.items()
        }
    )
