"""Clustering algorithms over the term graph."""

from __future__ import annotations

import random
from collections.abc import Iterator
from math import ceil

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import pymetis
import scipy.sparse as sp

from sharded_index.config import ClusteringMethod

METIS_WEIGHT_SCALE = 1000
"""METIS takes integer edge weights: the largest weight maps to this value."""

UNTIL_CONVERGENCE = -1
"""A negative iteration count makes leidenalg iterate until the partition stops improving."""


def _indexed_edges(edges: pd.DataFrame) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    """Nodes in first-appearance order and the edges as node indices with weights."""
    nodes = pd.Index(pd.unique(pd.concat([edges["src"], edges["dst"]], ignore_index=True)))
    node_id = {node: i for i, node in enumerate(nodes)}
    src = edges["src"].map(node_id).astype(np.int32).to_numpy()
    dst = edges["dst"].map(node_id).astype(np.int32).to_numpy()
    weights = np.maximum(edges["weight"].astype(float).to_numpy(), 0.0)
    return nodes, src, dst, weights


def _weighted_graph(
    n_nodes: int, src: np.ndarray, dst: np.ndarray, weights: np.ndarray
) -> ig.Graph:
    graph = ig.Graph(n=n_nodes, edges=list(zip(src, dst, strict=True)), directed=False)
    graph.es["weight"] = weights.tolist()
    return graph


def _metis(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
    n_parts: int,
    seed: int | None,
) -> list[int]:
    """K-way cut minimizing the weight of cut edges, parts balanced by node count."""
    adjacency = sp.coo_matrix(
        (np.r_[weights, weights], (np.r_[src, dst], np.r_[dst, src])),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    scale = METIS_WEIGHT_SCALE / max(float(adjacency.data.max()), 1e-12)
    edge_weights = np.maximum(np.rint(adjacency.data * scale), 1).astype(int)
    _, parts = pymetis.part_graph(
        min(max(n_parts, 2), n_nodes),
        adjacency=pymetis.CSRAdjacency(adjacency.indptr.tolist(), adjacency.indices.tolist()),
        eweights=edge_weights.tolist(),
        options=None if seed is None else pymetis.Options(seed=seed),
    )
    return list(parts)


def cluster_graph(
    edges: pd.DataFrame, method: ClusteringMethod, seed: int | None
) -> dict[str, int]:
    """Assign every graph term to a cluster.

    ``seed=None`` leaves the algorithm with its own default seeding.
    """
    if edges.empty:
        return {}
    nodes, src, dst, weights = _indexed_edges(edges)

    if method.algorithm == "metis":
        assert method.n_parts is not None
        membership = _metis(len(nodes), src, dst, weights, method.n_parts, seed)
    elif method.algorithm == "infomap":
        graph = _weighted_graph(len(nodes), src, dst, weights)
        ig.set_random_number_generator(random.Random(seed))
        membership = graph.community_infomap(edge_weights="weight").membership
    else:
        assert method.resolution is not None
        assert method.iterations is not None
        graph = _weighted_graph(len(nodes), src, dst, weights)
        quality = (
            la.RBConfigurationVertexPartition
            if method.algorithm == "leiden"
            else la.CPMVertexPartition
        )
        membership = la.find_partition(
            graph,
            quality,
            weights="weight",
            resolution_parameter=method.resolution,
            n_iterations=UNTIL_CONVERGENCE
            if method.iterations == "convergence"
            else method.iterations,
            seed=seed,
        ).membership
    return {str(node): int(cluster) for node, cluster in zip(nodes, membership, strict=True)}


def refinements(
    method: ClusteringMethod,
    volume_ratio: float,
    n_nodes: int,
    max_refinements: int,
) -> Iterator[ClusteringMethod]:
    """Progressively finer settings for splitting a cluster that exceeds a volume budget.

    Leiden and CPM double the resolution, METIS doubles the number of parts
    starting from ``volume_ratio`` (how many budgets the cluster holds);
    Infomap has no such parameter and is tried once.
    """
    if method.algorithm == "infomap":
        yield method
        return
    for step in range(max_refinements):
        if method.algorithm == "metis":
            n_parts = min(max(ceil(volume_ratio), 2) * 2**step, n_nodes)
            yield method.model_copy(update={"n_parts": n_parts})
        else:
            assert method.resolution is not None
            yield method.model_copy(update={"resolution": method.resolution * 2**step})


def modularity(edges: pd.DataFrame, clustering: dict[str, int]) -> float:
    """Weighted modularity of a clustering of the graph."""
    nodes, src, dst, weights = _indexed_edges(edges)
    graph = _weighted_graph(len(nodes), src, dst, weights)
    return float(graph.modularity([clustering[node] for node in nodes], weights="weight"))
