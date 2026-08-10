"""Tabular views of the Leiden term clusters."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .partition import TermPartition


def build_term_graph(partition: TermPartition) -> nx.Graph:
    """NetworkX view of the partition's NPMI co-occurrence graph."""
    return nx.from_pandas_edgelist(
        partition.edges_df, source="src", target="dst",
        edge_attr=["weight", "count"],
    )


def _terms_by_cluster(partition: TermPartition) -> dict[int, list[str]]:
    clusters: dict[int, list[str]] = {}
    for term, cluster_id in partition.term_to_shard.items():
        clusters.setdefault(cluster_id, []).append(term)
    return clusters


def top_terms_per_cluster(
    partition: TermPartition,
    *,
    n_clusters: int = 10,
    n_terms: int = 15,
) -> pd.DataFrame:
    """The strongest terms of the largest clusters.

    Returns a DataFrame ``[cluster, size, top_terms]`` for the ``n_clusters``
    largest clusters; terms are ranked by node strength.
    """
    clusters = _terms_by_cluster(partition)
    strength = partition.node_strength

    rows = []
    for cluster_id in partition.clusters_df["cluster"].head(n_clusters):
        terms = clusters.get(int(cluster_id), [])
        top = sorted(terms, key=lambda t: strength.get(t, 0.0), reverse=True)[:n_terms]
        rows.append({
            "cluster": int(cluster_id),
            "size": len(terms),
            "top_terms": ", ".join(top),
        })
    return pd.DataFrame(rows)


def cluster_summary_table(partition: TermPartition, *, top_n: int = 25) -> pd.DataFrame:
    """Per-cluster summary: size, strength, intra-cluster edges and density.

    ``density`` is the fraction of possible intra-cluster term pairs actually
    connected by an edge.
    """
    strength = partition.node_strength
    edges_df = partition.edges_df

    rows = []
    for cluster_id, terms in _terms_by_cluster(partition).items():
        strengths = [strength.get(t, 0.0) for t in terms]
        top = sorted(terms, key=lambda t: strength.get(t, 0.0), reverse=True)[:5]
        rows.append({
            "cluster": cluster_id,
            "size": len(terms),
            "total_strength": sum(strengths),
            "avg_strength": np.mean(strengths) if strengths else 0.0,
            "top_words": ", ".join(top),
        })
    summary = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)

    src_cluster = edges_df["src"].map(partition.term_to_shard)
    dst_cluster = edges_df["dst"].map(partition.term_to_shard)
    intra = src_cluster == dst_cluster

    if intra.any():
        intra_stats = (
            edges_df[intra]
            .assign(cluster=src_cluster[intra])
            .groupby("cluster")
            .agg(intra_edges=("weight", "size"), intra_weight=("weight", "sum"))
            .reset_index()
        )
        summary = summary.merge(intra_stats, on="cluster", how="left").fillna(0)
    else:
        summary["intra_edges"] = 0
        summary["intra_weight"] = 0.0

    possible_pairs = summary["size"] * (summary["size"] - 1) / 2
    summary["density"] = summary["intra_edges"] / (possible_pairs + 1e-9)
    return summary.head(top_n)
