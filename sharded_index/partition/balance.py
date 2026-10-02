"""Volume budget: re-partition terms into shards of bounded document volume.

A document lives in every shard of its terms, so the volume of a shard is the
number of documents holding at least one of its terms.  Clusters of frequent
terms can hold most of the corpus; here the number of shards follows from a
budget on that volume instead.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sharded_index.config import BalanceConfig, ClusteringMethod
from sharded_index.data.corpus import Corpus
from sharded_index.graph.clustering import cluster_graph, refinements
from sharded_index.partition.affinity import significant_choice


def balance(
    primary: Mapping[str, int],
    corpus: Corpus,
    edges: pd.DataFrame,
    method: ClusteringMethod,
    seed: int | None,
    config: BalanceConfig,
    hash_offset: int,
) -> dict[str, int]:
    """Re-partition the vocabulary under the volume budget ``volume_cap * n_docs``.

    1. Units.  A cluster within the budget is one unit.  A cluster above it
       is split: its graph terms into communities by ``method`` run on the
       cluster's own edges, and each out-of-graph term joins the community it
       is most significantly over-represented with in documents.  A unit still
       above the budget is split again; one without graph structure left
       falls apart into single terms.  Terms of the hash space
       (``>= hash_offset``) are single-term units.
    2. Packing.  Units, largest first, go to the shard whose volume they
       increase least among the shards that stay within the budget; a new
       shard opens only when none fits.

    A term more frequent than the budget ends up in a shard of its own.
    Shard ids are ``0..S-1`` in packing order.
    """
    if set(primary) != set(corpus.vocabulary):
        msg = "the partition must cover exactly the corpus vocabulary"
        raise ValueError(msg)
    return _Balancer(primary, corpus, edges, method, seed, config, hash_offset).run()


class _Balancer:
    """State of one balancing run; terms are addressed by their vocabulary column."""

    def __init__(
        self,
        primary: Mapping[str, int],
        corpus: Corpus,
        edges: pd.DataFrame,
        method: ClusteringMethod,
        seed: int | None,
        config: BalanceConfig,
        hash_offset: int,
    ) -> None:
        self.vocabulary = corpus.vocabulary
        self.by_term = corpus.by_term
        self.n_docs = corpus.n_docs
        self.df = corpus.df.astype(np.float64)
        self.cap = config.volume_cap * corpus.n_docs
        self.method = method
        self.seed = seed
        self.config = config

        self.column = {term: j for j, term in enumerate(self.vocabulary)}
        self.shard_of = np.array([primary[term] for term in self.vocabulary], dtype=np.int64)
        self.in_hash_space = self.shard_of >= hash_offset

        src = edges["src"].map(self.column)
        dst = edges["dst"].map(self.column)
        in_vocabulary = src.notna() & dst.notna()
        self.edge_src = src[in_vocabulary].to_numpy(dtype=np.int64)
        self.edge_dst = dst[in_vocabulary].to_numpy(dtype=np.int64)
        self.edge_weight = edges["weight"][in_vocabulary].to_numpy(dtype=np.float64)
        self.is_graph_term = np.zeros(len(self.vocabulary), dtype=bool)
        self.is_graph_term[self.edge_src] = True
        self.is_graph_term[self.edge_dst] = True

    def run(self) -> dict[str, int]:
        units = [np.array([j]) for j in np.flatnonzero(self.in_hash_space)]
        volumes = self._cluster_volumes()
        for cluster in np.unique(self.shard_of[~self.in_hash_space]):
            columns = np.flatnonzero(self.shard_of == cluster)
            if volumes[cluster] <= self.cap:
                units.append(columns)
                continue
            graph = columns[self.is_graph_term[columns]]
            tail = columns[~self.is_graph_term[columns]]
            shared_docs = (self.by_term[:, tail].T @ self.by_term[:, graph]).tocsr()
            units += self._split(graph, tail, shared_docs, depth=0)

        shard = np.empty(len(self.vocabulary), dtype=np.int64)
        for shard_id, columns in enumerate(self._pack(units)):
            shard[columns] = shard_id
        return dict(zip(self.vocabulary, shard.tolist(), strict=True))

    def _cluster_volumes(self) -> np.ndarray:
        """Document volume of every cluster of the input partition."""
        n_terms = len(self.vocabulary)
        term_cluster = sp.csr_matrix(
            (np.ones(n_terms), (np.arange(n_terms), self.shard_of)),
            shape=(n_terms, int(self.shard_of.max(initial=-1)) + 1),
        )
        return np.asarray(((self.by_term @ term_cluster) > 0).sum(axis=0)).ravel()

    def _documents(self, columns: np.ndarray) -> np.ndarray:
        """Documents holding at least one of the terms."""
        return np.unique(self.by_term[:, columns].indices)

    def _communities(self, graph: np.ndarray, volume: int) -> list[np.ndarray] | None:
        """Communities of the terms' own subgraph; ``None`` if it stays in one piece."""
        inside = np.isin(self.edge_src, graph) & np.isin(self.edge_dst, graph)
        if not inside.any():
            return None
        subgraph = pd.DataFrame(
            {
                "src": [self.vocabulary[j] for j in self.edge_src[inside]],
                "dst": [self.vocabulary[j] for j in self.edge_dst[inside]],
                "weight": self.edge_weight[inside],
            }
        )
        n_nodes = len(np.union1d(self.edge_src[inside], self.edge_dst[inside]))
        attempts = refinements(self.method, volume / self.cap, n_nodes, self.config.max_refinements)
        for refined in attempts:
            groups: dict[int, list[int]] = defaultdict(list)
            for term, community in cluster_graph(subgraph, refined, self.seed).items():
                groups[community].append(self.column[term])
            clustered = {j for group in groups.values() for j in group}
            isolated = sorted(set(graph.tolist()) - clustered)
            if len(groups) + len(isolated) > 1:
                communities = [np.array(sorted(group)) for _, group in sorted(groups.items())]
                return communities + [np.array([j]) for j in isolated]
        return None

    def _split(
        self,
        graph: np.ndarray,
        tail: np.ndarray,
        shared_docs: sp.csr_matrix,
        depth: int,
    ) -> list[np.ndarray]:
        """Split a unit of graph and tail terms until every piece fits the budget.

        ``shared_docs[i, j]`` counts the documents shared by ``tail[i]`` and ``graph[j]``.
        """
        columns = np.concatenate([graph, tail])
        volume = len(self._documents(columns)) if len(columns) > 1 else 0
        if volume <= self.cap:
            return [columns]
        can_split = depth < self.config.max_depth and len(graph) > 1
        communities = self._communities(graph, volume) if can_split else None
        if communities is None:
            return [np.array([j]) for j in columns]

        position = {j: i for i, j in enumerate(graph.tolist())}
        group = np.empty(len(graph), dtype=np.int64)
        for g, community in enumerate(communities):
            group[[position[j] for j in community.tolist()]] = g
        term_group = sp.csr_matrix(
            (np.ones(len(graph)), (np.arange(len(graph)), group)),
            shape=(len(graph), len(communities)),
        )
        mass = np.bincount(group, weights=self.df[graph], minlength=len(communities))
        choice = significant_choice(shared_docs @ term_group, mass, "llr")

        pieces: list[np.ndarray] = []
        for g in range(len(communities)):
            graph_rows = np.flatnonzero(group == g)
            tail_rows = np.flatnonzero(choice == g)
            pieces += self._split(
                graph[graph_rows],
                tail[tail_rows],
                shared_docs[tail_rows][:, graph_rows],
                depth + 1,
            )
        pieces += [np.array([j]) for j in tail[choice < 0]]
        return pieces

    def _pack(self, units: list[np.ndarray]) -> list[np.ndarray]:
        """Best-fit packing of units into shards by volume increment under the budget."""
        unit_docs = [self._documents(unit) for unit in units]
        order = sorted(range(len(units)), key=lambda i: (-len(unit_docs[i]), int(units[i].min())))
        shard_docs: list[np.ndarray] = []
        volumes: list[int] = []
        members: list[list[np.ndarray]] = []
        for i in order:
            docs = unit_docs[i]
            best, best_increment = -1, 0
            for shard, present in enumerate(shard_docs):
                increment = len(docs) - int(present[docs].sum())
                fits = volumes[shard] + increment <= self.cap
                if fits and (best < 0 or increment < best_increment):
                    best, best_increment = shard, increment
            if best < 0:
                shard_docs.append(np.zeros(self.n_docs, dtype=bool))
                volumes.append(0)
                members.append([])
                best, best_increment = len(shard_docs) - 1, len(docs)
            shard_docs[best][docs] = True
            volumes[best] += best_increment
            members[best].append(units[i])
        return [np.concatenate(parts) for parts in members]
