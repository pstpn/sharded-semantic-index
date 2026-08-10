"""Leiden community detection over the term co-occurrence graph."""

from __future__ import annotations

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd


def cluster_terms(
    edges_df: pd.DataFrame,
    *,
    resolution: float = 1.0,
    seed: int = 42,
    method: str = "leiden",
) -> tuple[dict[str, int], pd.DataFrame]:
    """Partition graph terms into clusters.

    Parameters
    ----------
    edges_df:
        Edge list with columns ``[src, dst, weight]``.
    resolution:
        Leiden/CPM resolution parameter (higher → more, smaller clusters);
        for ``method="metis"`` it is reinterpreted as the number of parts.
    seed:
        Random seed for reproducibility.
    method:
        ``leiden`` (modularity, RBConfiguration), ``cpm``
        (no resolution limit), ``infomap`` (flow-based objective) or
        ``metis`` (balanced k-way; needs the optional ``pymetis`` package).

    Returns
    -------
    term_to_cluster : dict[str, int]
        Mapping from term to cluster id.
    clusters_df : pd.DataFrame
        Columns ``[cluster, size]``, sorted by size descending.
    """
    if edges_df.empty:
        return {}, pd.DataFrame(columns=["cluster", "size"])

    nodes = pd.Index(pd.unique(
        pd.concat([edges_df["src"], edges_df["dst"]], ignore_index=True)
    ))
    node_to_id = {n: i for i, n in enumerate(nodes)}

    src_ids = edges_df["src"].map(node_to_id).astype(np.int32).to_numpy()
    dst_ids = edges_df["dst"].map(node_to_id).astype(np.int32).to_numpy()
    weights = np.maximum(edges_df["weight"].astype(float).to_numpy(), 0.0)

    graph = ig.Graph(n=len(nodes), edges=list(zip(src_ids, dst_ids)), directed=False)
    graph.es["weight"] = weights.tolist()

    if method in ("leiden", "cpm"):
        kind = (
            la.RBConfigurationVertexPartition if method == "leiden"
            else la.CPMVertexPartition
        )
        part = la.find_partition(
            graph, kind,
            weights="weight",
            resolution_parameter=float(resolution),
            seed=int(seed),
        )
        membership = np.asarray(part.membership, dtype=np.int32)
    elif method == "infomap":
        import random as _random

        _random.seed(int(seed))
        ig.set_random_number_generator(_random)
        part = graph.community_infomap(edge_weights="weight")
        membership = np.asarray(part.membership, dtype=np.int32)
    elif method == "metis":
        try:
            import pymetis
        except ImportError as exc:  # pragma: no cover
            raise ValueError(
                "method='metis' needs the optional pymetis package: pip install pymetis"
            ) from exc
        adjacency: list[list[int]] = [[] for _ in range(len(nodes))]
        for a, b in zip(src_ids, dst_ids):
            adjacency[a].append(int(b))
            adjacency[b].append(int(a))
        _, parts = pymetis.part_graph(max(int(resolution), 2), adjacency=adjacency)
        membership = np.asarray(parts, dtype=np.int32)
    else:
        raise ValueError(f"unknown clustering method: {method!r}")
    term_to_cluster = {str(nodes[i]): int(membership[i]) for i in range(len(nodes))}

    sizes = pd.Series(membership).value_counts().sort_values(ascending=False)
    clusters_df = pd.DataFrame({
        "cluster": sizes.index.astype(int),
        "size": sizes.values.astype(int),
    })
    return term_to_cluster, clusters_df
