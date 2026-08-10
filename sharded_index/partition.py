"""Term-to-shard partition and routing.

:class:`TermPartition` is the heart of the project: it assigns every term to a
shard and routes queries/documents to **all** shards containing their terms
(full routing, not a majority-vote single shard).  Two construction
strategies are compared in the benchmarks:

- :meth:`TermPartition.from_corpus` — semantic: Leiden clusters of the NPMI graph;
- :meth:`TermPartition.from_hashing` — baseline: ``hash(term) % n_shards`` (Maguro-style).
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clustering import cluster_terms
from .config import PartitionConfig
from .graph import EDGE_COLUMNS, build_npmi_graph, compute_node_strength
from .text import tokenize


def hash_term_to_shard(term: str, n_shards: int) -> int:
    """Deterministic hash-based shard for a term (MD5, stable across runs)."""
    return int(hashlib.md5(term.encode()).hexdigest(), 16) % n_shards


def cluster_affinity(edges_df: pd.DataFrame, term_to_shard: dict[str, int]) -> pd.DataFrame:
    """Affinity of every graph term to every cluster it has edges into.

    Affinity(t, c) = sum of NPMI weights of edges from ``t`` to terms of
    cluster ``c``.  A term's own cluster is normally its top-affinity one;
    the next clusters are natural candidates for replication.

    Returns a DataFrame ``[term, cluster, affinity]``.
    """
    if edges_df.empty:
        return pd.DataFrame(columns=["term", "cluster", "affinity"])

    directed = pd.concat([
        pd.DataFrame({
            "term": edges_df["src"],
            "cluster": edges_df["dst"].map(term_to_shard),
            "affinity": edges_df["weight"],
        }),
        pd.DataFrame({
            "term": edges_df["dst"],
            "cluster": edges_df["src"].map(term_to_shard),
            "affinity": edges_df["weight"],
        }),
    ], ignore_index=True)
    return directed.groupby(["term", "cluster"], as_index=False)["affinity"].sum()


def _greedy_cover(
    term_candidates: dict[str, tuple[int, ...]],
    strength: dict[str, float],
    max_shards: int | None = None,
) -> list[int]:
    """Greedy weighted set cover: an ordered list of shards covering all terms.

    Each step picks the shard whose still-uncovered terms carry the largest
    total node strength (ties → smallest shard id, deterministic).  With
    single-assignment candidates this degenerates into sorting shards by the
    total strength of their terms.
    """
    uncovered = set(term_candidates)
    order: list[int] = []
    while uncovered and (max_shards is None or len(order) < max_shards):
        gains: dict[int, float] = defaultdict(float)
        for term in uncovered:
            for shard in term_candidates[term]:
                gains[shard] += strength.get(term, 0.0)
        best = min(gains, key=lambda s: (-gains[s], s))
        order.append(best)
        uncovered = {t for t in uncovered if best not in term_candidates[t]}
    return order


class TermPartition:
    """Term → shard mapping with query/document routing.

    Usage::

        part = TermPartition.from_corpus(texts, config=PartitionConfig())
        part.shards_for_query("how to reset windows password")
        # {3, 8, 5} — ALL shards holding any of the query terms
    """

    def __init__(
        self,
        term_to_shard: dict[str, int],
        node_strength: dict[str, float] | None = None,
        edges_df: pd.DataFrame | None = None,
        clusters_df: pd.DataFrame | None = None,
    ):
        self.term_to_shard = dict(term_to_shard)
        self.node_strength = node_strength or {}
        self.edges_df = edges_df if edges_df is not None else pd.DataFrame(columns=EDGE_COLUMNS)
        self.clusters_df = (
            clusters_df if clusters_df is not None else pd.DataFrame(columns=["cluster", "size"])
        )

    # ── Construction ──────────────────────────────────────────────────

    @classmethod
    def from_corpus(
        cls,
        texts: list[str],
        *,
        config: PartitionConfig | None = None,
    ) -> TermPartition:
        """Semantic partition: NPMI co-occurrence graph → Leiden clusters.

        Covers only terms that made it into the graph; extend to a full
        vocabulary with :meth:`with_hash_fallback`.
        """
        config = config or PartitionConfig()
        edges_df = build_npmi_graph(
            texts,
            min_df=config.min_df,
            max_df_ratio=config.max_df_ratio,
            min_pair_count=config.min_pair_count,
            min_npmi=config.min_npmi,
            edge_weight=config.edge_weight,
            extra_stop_words=tuple(config.graph_stop_words),
        )
        term_to_shard, clusters_df = cluster_terms(
            edges_df,
            resolution=config.leiden_resolution,
            seed=config.leiden_seed,
            method=config.cluster_method,
        )
        return cls(term_to_shard, compute_node_strength(edges_df), edges_df, clusters_df)

    @classmethod
    def from_hashing(
        cls,
        terms: Iterable[str],
        n_shards: int,
        *,
        node_strength: dict[str, float] | None = None,
    ) -> TermPartition:
        """Baseline partition: ``hash(term) % n_shards`` for every term.

        ``node_strength`` may be borrowed from a semantic partition so that
        ranked routing stays comparable between strategies.
        """
        # sorted() — только ради байт-стабильных артефактов; значения не зависят
        # от порядка.
        return cls(
            {t: hash_term_to_shard(t, n_shards) for t in sorted(terms)},
            node_strength,
        )

    def with_hash_fallback(self, terms: Iterable[str]) -> TermPartition:
        """Extend the partition to cover ``terms`` (typically the corpus vocabulary).

        Terms already in the partition keep their cluster; unknown terms are
        hashed into a **disjoint** fallback shard space ``[n_shards, 2 * n_shards)``,
        so semantic shards stay purely semantic.  Terms of the original
        partition absent from ``terms`` are dropped.
        """
        n = self.n_shards
        term_to_shard = {
            t: self.term_to_shard.get(t, n + hash_term_to_shard(t, n))
            for t in sorted(terms)
        }
        return TermPartition(term_to_shard, self.node_strength, self.edges_df, self.clusters_df)

    def with_volume_cap(
        self,
        doc_token_sets: Iterable[Collection[str]],
        term_df: dict[str, int],
        *,
        max_volume_ratio: float,
        resolution: float = 1.0,
        seed: int = 42,
        growth: float = 4.0,
        max_rounds: int = 6,
        fallback_offset: int | None = None,
    ) -> TermPartition:
        """Split clusters whose *document volume* exceeds a budget.

        Modularity clustering leaves a giant cluster of frequent terms; since
        a document replicates to all shards of its terms, that cluster's shard
        accumulates almost the whole corpus and "one probed shard" stops being
        cheap.  This post-processing re-runs Leiden with growing resolution on
        the subgraph of every oversized cluster (targeted: healthy topical
        clusters stay intact).  A cluster whose subgraph refuses to split is
        packed greedily by document frequency into ``ceil(volume / cap)`` bins.

        The volume of a shard is bounded from below by the largest df of its
        terms, so a cluster whose top term alone exceeds the budget is marked
        unsplittable and kept as is — the reachable budget floor is
        ``max(term_df) / n_docs``.

        Parameters
        ----------
        doc_token_sets:
            Tokenized corpus documents; volumes are measured against them.
        term_df:
            Document frequency per term (for the packing fallback and floor).
        max_volume_ratio:
            Volume budget per shard as a share of the corpus.
        resolution, seed:
            Leiden parameters of the base clustering; splits use
            ``resolution * growth ** round``.
        fallback_offset:
            When the partition already contains a hash-fallback space (ids
            ``>= fallback_offset``), pass the offset: split shards get fresh
            ids beyond it, so afterwards all ids are renumbered — semantic
            clusters to ``[0, K)``, hash-space terms to ``[K, K + H)`` — to
            keep the "semantic below fallback" id contract of the pipeline
            diagnostics.
        """
        from .clustering import cluster_terms

        doc_tokens = [set(t) for t in doc_token_sets]
        n_docs = len(doc_tokens)
        cap_docs = max_volume_ratio * n_docs
        term_to_shard = dict(self.term_to_shard)
        next_id = max(term_to_shard.values(), default=-1) + 1
        unsplittable: set[int] = set()
        hash_ids: set[int] = (
            {s for s in term_to_shard.values() if s >= fallback_offset}
            if fallback_offset is not None else set()
        )

        for round_no in range(max_rounds):
            volumes: Counter[int] = Counter()
            for tokens in doc_tokens:
                clusters = {term_to_shard[t] for t in tokens if t in term_to_shard}
                volumes.update(clusters)

            oversized = [
                c for c, v in volumes.items()
                if v > cap_docs and c not in unsplittable
            ]
            if not oversized:
                break

            terms_by_cluster: dict[int, list[str]] = defaultdict(list)
            for t, c in term_to_shard.items():
                terms_by_cluster[c].append(t)

            src_cluster = self.edges_df["src"].map(term_to_shard)
            dst_cluster = self.edges_df["dst"].map(term_to_shard)

            for cid in sorted(oversized):
                members = terms_by_cluster[cid]
                # Хеш-бины не расщепляются: это резерв для термов-сирот.
                if cid in hash_ids:
                    unsplittable.add(cid)
                    continue
                # Пол объёма: шард не меньше df самого частого своего терма.
                # Расщепление бессмысленно, лишь когда объём кластера уже
                # исчерпывается этим термом (или терм единственный).
                dominant_df = max(term_df.get(t, 0) for t in members)
                if len(members) <= 1 or volumes[cid] <= dominant_df * 1.05:
                    unsplittable.add(cid)
                    continue

                sub_edges = self.edges_df[(src_cluster == cid) & (dst_cluster == cid)]
                sub_map, _ = cluster_terms(
                    sub_edges,
                    resolution=resolution * growth ** (round_no + 1),
                    seed=seed,
                )
                parts = set(sub_map.values())
                if len(parts) > 1:
                    id_map = {p: next_id + i for i, p in enumerate(sorted(parts))}
                    next_id += len(parts)
                    for t in members:
                        if t in sub_map:
                            term_to_shard[t] = id_map[sub_map[t]]
                    # Термы без внутренних рёбер: к наименее нагруженной части.
                    loads = Counter()
                    for t in members:
                        if t in sub_map:
                            loads[term_to_shard[t]] += term_df.get(t, 0)
                    for t in sorted(set(members) - set(sub_map)):
                        best = min(id_map.values(), key=lambda b: (loads[b], b))
                        term_to_shard[t] = best
                        loads[best] += term_df.get(t, 0)
                else:
                    # Граф не делится (клика/нет рёбер) — df-жадная упаковка.
                    n_bins = max(int(-(-volumes[cid] // cap_docs)), 2)
                    bins = [next_id + i for i in range(n_bins)]
                    next_id += n_bins
                    loads = Counter({b: 0 for b in bins})
                    for t in sorted(members, key=lambda t: (-term_df.get(t, 0), t)):
                        best = min(bins, key=lambda b: (loads[b], b))
                        term_to_shard[t] = best
                        loads[best] += term_df.get(t, 0)

        if fallback_offset is not None:
            # Семантические кластеры (включая новые части) → [0, K),
            # хеш-пространство → [K, K + H): контракт id для диагностик.
            semantic_ids = sorted(set(term_to_shard.values()) - hash_ids)
            remap = {c: i for i, c in enumerate(semantic_ids)}
            k = len(semantic_ids)
            for i, c in enumerate(sorted(set(term_to_shard.values()) & hash_ids)):
                remap[c] = k + i
            term_to_shard = {t: remap[c] for t, c in term_to_shard.items()}

        sizes = pd.Series(list(term_to_shard.values())).value_counts()
        clusters_df = pd.DataFrame({
            "cluster": sizes.index.astype(int),
            "size": sizes.values.astype(int),
        }).sort_values("size", ascending=False, ignore_index=True)
        return TermPartition(term_to_shard, self.node_strength, self.edges_df, clusters_df)

    def with_affinity_fallback(
        self,
        doc_token_sets: Iterable[Collection[str]],
    ) -> TermPartition:
        """Extend the partition to the corpus vocabulary via document co-occurrence.

        Terms already in the partition keep their cluster.  An out-of-graph
        term is assigned to the semantic cluster it co-occurs with most: every
        document votes with the cluster multiset of its graph terms (ties →
        smaller cluster id).  Only terms that never share a document with any
        graph term fall into the disjoint hash space ``[n_shards, 2 * n_shards)``.

        This removes the structural weakness of :meth:`with_hash_fallback`,
        where a query mixing a semantic and a rare term always needs at least
        two shards because the two spaces are disjoint.

        Parameters
        ----------
        doc_token_sets:
            Tokenized corpus documents (e.g. ``set(tokenize(text))`` per doc);
            their union defines the vocabulary to cover.
        """
        n = self.n_shards
        votes: dict[str, Counter[int]] = defaultdict(Counter)
        vocab: set[str] = set()

        for tokens in doc_token_sets:
            vocab.update(tokens)
            doc_clusters = Counter(
                self.term_to_shard[t] for t in tokens if t in self.term_to_shard
            )
            if not doc_clusters:
                continue
            for t in tokens:
                if t not in self.term_to_shard:
                    votes[t].update(doc_clusters)

        term_to_shard: dict[str, int] = {}
        for t in sorted(vocab):
            known = self.term_to_shard.get(t)
            term_votes = votes.get(t)
            if known is not None:
                term_to_shard[t] = known
            elif term_votes:
                # argmax голосов, при равенстве — меньший id кластера
                term_to_shard[t] = min(term_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            else:
                term_to_shard[t] = n + hash_term_to_shard(t, n)
        return TermPartition(term_to_shard, self.node_strength, self.edges_df, self.clusters_df)

    # ── Persistence (parquet files, used by the DVC pipeline) ─────────

    def save(self, target_dir: Path) -> None:
        """Persist the partition to a directory of parquet files.

        Empty optional parts (edges, clusters, strengths) are simply not
        written; :meth:`load` restores them as empty defaults.
        """
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "term": list(self.term_to_shard.keys()),
            "shard": list(self.term_to_shard.values()),
        }).to_parquet(target_dir / "term_to_shard.parquet", index=False)

        if self.node_strength:
            pd.DataFrame({
                "term": list(self.node_strength.keys()),
                "strength": list(self.node_strength.values()),
            }).to_parquet(target_dir / "node_strength.parquet", index=False)
        if not self.edges_df.empty:
            self.edges_df.to_parquet(target_dir / "edges.parquet", index=False)
        if not self.clusters_df.empty:
            self.clusters_df.to_parquet(target_dir / "clusters.parquet", index=False)

    @classmethod
    def load(cls, source_dir: Path) -> TermPartition:
        """Load a partition saved by :meth:`save` (row order is preserved)."""
        source_dir = Path(source_dir)

        mapping = pd.read_parquet(source_dir / "term_to_shard.parquet")
        term_to_shard = dict(zip(mapping["term"].tolist(), mapping["shard"].tolist()))

        strength_path = source_dir / "node_strength.parquet"
        node_strength = None
        if strength_path.exists():
            strength = pd.read_parquet(strength_path)
            node_strength = dict(zip(strength["term"].tolist(), strength["strength"].tolist()))

        edges_path = source_dir / "edges.parquet"
        clusters_path = source_dir / "clusters.parquet"
        return cls(
            term_to_shard,
            node_strength,
            pd.read_parquet(edges_path) if edges_path.exists() else None,
            pd.read_parquet(clusters_path) if clusters_path.exists() else None,
        )

    # ── Basic properties ──────────────────────────────────────────────

    @property
    def n_shards(self) -> int:
        """Number of shards (``max shard id + 1``)."""
        return max(self.term_to_shard.values(), default=-1) + 1

    @property
    def n_terms(self) -> int:
        """Number of terms with a shard assignment."""
        return len(self.term_to_shard)

    # ── Routing ───────────────────────────────────────────────────────

    def shards_for_tokens(self, tokens: Iterable[str]) -> set[int]:
        """All shards containing any of the given tokens."""
        return {self.term_to_shard[t] for t in tokens if t in self.term_to_shard}

    def shards_for_query(self, query: str) -> set[int]:
        """All shards for all query terms."""
        return self.shards_for_tokens(tokenize(query))

    def shards_for_document(self, doc_text: str) -> set[int]:
        """All shards for all document terms."""
        return self.shards_for_tokens(tokenize(doc_text))

    def ranked_shards_for_query(self, query: str) -> list[int]:
        """Query shards ordered by aggregate node strength of matching terms.

        Shards whose query terms carry more total NPMI strength come first —
        useful for early-termination strategies that probe only top-k shards.
        """
        shards = self.shards_for_query(query)
        if not shards:
            return []

        strength_by_shard: dict[int, float] = defaultdict(float)
        for token in tokenize(query):
            if token in self.term_to_shard and token in self.node_strength:
                strength_by_shard[self.term_to_shard[token]] += self.node_strength[token]

        return sorted(shards, key=lambda s: strength_by_shard.get(s, 0.0), reverse=True)

    def cover_shards_for_query(self, query: str, max_shards: int | None = None) -> list[int]:
        """Greedy cover of the query terms by shards (see :func:`_greedy_cover`).

        With single assignment every term forces its one shard, so the cover
        is all query shards ordered by aggregate strength — the same order
        as :meth:`ranked_shards_for_query` up to ties.
        """
        tokens = set(tokenize(query))
        candidates = {
            t: (self.term_to_shard[t],) for t in tokens if t in self.term_to_shard
        }
        return _greedy_cover(candidates, self.node_strength, max_shards)

    # ── Statistics ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Summary statistics of shard sizes (in terms)."""
        if not self.term_to_shard:
            return {"n_shards": 0, "n_terms": 0}

        sizes = pd.Series(list(self.term_to_shard.values())).value_counts()
        return {
            "n_shards": self.n_shards,
            "n_terms": self.n_terms,
            "shard_size_min": int(sizes.min()),
            "shard_size_max": int(sizes.max()),
            "shard_size_mean": float(sizes.mean()),
            "shard_size_median": float(np.median(sizes.values)),
        }


def assign_docs_to_shards(
    docs: dict[str, str],
    partition: TermPartition,
) -> tuple[dict[str, set[int]], dict[int, dict[str, str]]]:
    """Assign every document to **all** shards where its terms live.

    Parameters
    ----------
    docs:
        Mapping ``doc_id → doc_text``.
    partition:
        Term-to-shard partition used for routing.

    Returns
    -------
    doc_to_shards : dict[str, set[int]]
        ``doc_id → set of shard ids`` (used by routing-recall evaluation).
    docs_by_shard : dict[int, dict[str, str]]
        ``shard_id → {doc_id: doc_text}`` (input for index building).
    """
    doc_to_shards: dict[str, set[int]] = {}
    docs_by_shard: dict[int, dict[str, str]] = defaultdict(dict)

    for doc_id, doc_text in docs.items():
        shards = partition.shards_for_document(doc_text)
        doc_to_shards[doc_id] = shards
        for shard_id in shards:
            docs_by_shard[shard_id][doc_id] = doc_text

    return doc_to_shards, dict(docs_by_shard)


class ReplicatedTermPartition:
    """Term → several shards: the primary cluster plus top-affinity neighbors.

    The research trade-off behind it: documents replicate to all shards of
    all their terms, so duplication grows with the replica count — but a
    query can now be *covered* by fewer shards (down to a single one when
    some shard contains every query term), instead of probing one shard per
    distinct term cluster.

    Only graph terms get extra replicas (affinity is defined by NPMI edges);
    hash-fallback terms keep their single shard.
    """

    def __init__(
        self,
        term_to_shards: dict[str, tuple[int, ...]],
        node_strength: dict[str, float] | None = None,
        edges_df: pd.DataFrame | None = None,
        clusters_df: pd.DataFrame | None = None,
    ):
        self.term_to_shards = {t: tuple(s) for t, s in term_to_shards.items()}
        self.node_strength = node_strength or {}
        self.edges_df = edges_df if edges_df is not None else pd.DataFrame(columns=EDGE_COLUMNS)
        self.clusters_df = (
            clusters_df if clusters_df is not None else pd.DataFrame(columns=["cluster", "size"])
        )

    @classmethod
    def from_partition(
        cls,
        base: TermPartition,
        n_replicas: int,
        *,
        affinity_quantile: float | None = None,
    ) -> ReplicatedTermPartition:
        """Replicate graph terms of ``base`` into their ``n_replicas`` best clusters.

        The first shard of every term is its ``base`` assignment (so
        ``n_replicas=1`` reproduces ``base`` exactly); the extras are the
        clusters with the highest :func:`cluster_affinity`, semantic space
        only.  ``base`` is normally a partition already extended with
        :meth:`TermPartition.with_hash_fallback`.

        ``affinity_quantile`` (0..1) keeps only extra candidates whose
        affinity is at or above that quantile of all inter-cluster
        affinities — a cost control: replicas concentrate on the strongest
        "bridge" terms instead of everything with a cross-cluster edge.
        """
        if n_replicas < 1:
            raise ValueError("n_replicas must be >= 1")

        affinity = cluster_affinity(base.edges_df, base.term_to_shard)
        ranked_clusters: dict[str, list[int]] = {}
        if not affinity.empty:
            # Кандидаты на реплики — только межкластерные связи терма.
            extra = affinity[affinity["cluster"] != affinity["term"].map(base.term_to_shard)]
            if affinity_quantile is not None and not extra.empty:
                threshold = extra["affinity"].quantile(affinity_quantile)
                extra = extra[extra["affinity"] >= threshold]
            if not extra.empty:
                extra = extra.sort_values(
                    ["term", "affinity", "cluster"], ascending=[True, False, True], kind="stable",
                )
                ranked_clusters = extra.groupby("term", sort=False)["cluster"].agg(list).to_dict()

        term_to_shards = {}
        for term, primary in base.term_to_shard.items():
            extras = [int(c) for c in ranked_clusters.get(term, [])]
            term_to_shards[term] = (primary, *extras[: n_replicas - 1])
        return cls(term_to_shards, base.node_strength, base.edges_df, base.clusters_df)

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, target_dir: Path) -> None:
        """Persist to parquet files (loadable via :func:`load_partition`)."""
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        rows = [
            (term, shard, rank)
            for term, shards in self.term_to_shards.items()
            for rank, shard in enumerate(shards)
        ]
        pd.DataFrame(rows, columns=["term", "shard", "rank"]).to_parquet(
            target_dir / "term_to_shards.parquet", index=False,
        )
        if self.node_strength:
            pd.DataFrame({
                "term": list(self.node_strength.keys()),
                "strength": list(self.node_strength.values()),
            }).to_parquet(target_dir / "node_strength.parquet", index=False)
        if not self.edges_df.empty:
            self.edges_df.to_parquet(target_dir / "edges.parquet", index=False)
        if not self.clusters_df.empty:
            self.clusters_df.to_parquet(target_dir / "clusters.parquet", index=False)

    @classmethod
    def load(cls, source_dir: Path) -> ReplicatedTermPartition:
        source_dir = Path(source_dir)
        rows = pd.read_parquet(source_dir / "term_to_shards.parquet")

        term_to_shards: dict[str, list[int]] = {}
        for term, shard in zip(rows["term"].tolist(), rows["shard"].tolist()):
            term_to_shards.setdefault(term, []).append(shard)  # rows are rank-ordered

        strength_path = source_dir / "node_strength.parquet"
        node_strength = None
        if strength_path.exists():
            strength = pd.read_parquet(strength_path)
            node_strength = dict(zip(strength["term"].tolist(), strength["strength"].tolist()))

        edges_path = source_dir / "edges.parquet"
        clusters_path = source_dir / "clusters.parquet"
        return cls(
            {t: tuple(s) for t, s in term_to_shards.items()},
            node_strength,
            pd.read_parquet(edges_path) if edges_path.exists() else None,
            pd.read_parquet(clusters_path) if clusters_path.exists() else None,
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def n_shards(self) -> int:
        """Number of shards (``max shard id + 1``)."""
        return max((max(s) for s in self.term_to_shards.values()), default=-1) + 1

    @property
    def n_terms(self) -> int:
        return len(self.term_to_shards)

    @property
    def replication_factor(self) -> float:
        """Mean number of shards per term (1.0 = no replication)."""
        if not self.term_to_shards:
            return 0.0
        return sum(len(s) for s in self.term_to_shards.values()) / len(self.term_to_shards)

    # ── Routing ───────────────────────────────────────────────────────

    def shards_for_tokens(self, tokens: Iterable[str]) -> set[int]:
        """All shards of all replicas of the given tokens."""
        return {
            shard
            for t in tokens
            if t in self.term_to_shards
            for shard in self.term_to_shards[t]
        }

    def shards_for_query(self, query: str) -> set[int]:
        return self.shards_for_tokens(tokenize(query))

    def shards_for_document(self, doc_text: str) -> set[int]:
        return self.shards_for_tokens(tokenize(doc_text))

    def cover_shards_for_query(self, query: str, max_shards: int | None = None) -> list[int]:
        """Greedy cover of the query terms — the point of replication.

        With replicas, co-occurring terms often share a cluster, so the
        whole query fits into one or two shards.
        """
        tokens = set(tokenize(query))
        candidates = {t: self.term_to_shards[t] for t in tokens if t in self.term_to_shards}
        return _greedy_cover(candidates, self.node_strength, max_shards)


def replicate_with_volume_guard(
    base: TermPartition,
    n_replicas: int,
    doc_token_sets: Iterable[Collection[str]],
    term_df: dict[str, int],
    *,
    max_volume_ratio: float,
    affinity_quantile: float | None = None,
) -> ReplicatedTermPartition:
    """Term replication that respects the shard volume budget.

    Same as :meth:`ReplicatedTermPartition.from_partition`, but a replica of
    term ``t`` is admitted into shard ``s`` only while the shard's measured
    document volume plus the df of replicas already admitted there stays
    within ``max_volume_ratio`` of the corpus.  The df sum overestimates the
    true volume growth (documents overlap), so the guard is conservative.
    Deterministic: terms are processed in sorted order.
    """
    doc_tokens = [set(t) for t in doc_token_sets]
    cap_docs = max_volume_ratio * len(doc_tokens)
    volumes: Counter[int] = Counter()
    for tokens in doc_tokens:
        volumes.update(base.shards_for_tokens(tokens))

    raw = ReplicatedTermPartition.from_partition(
        base, n_replicas, affinity_quantile=affinity_quantile,
    )
    added: Counter[int] = Counter()
    guarded: dict[str, tuple[int, ...]] = {}
    for term in sorted(raw.term_to_shards):
        shards = raw.term_to_shards[term]
        keep = [shards[0]]
        for shard in shards[1:]:
            if volumes[shard] + added[shard] + term_df.get(term, 0) <= cap_docs:
                keep.append(shard)
                added[shard] += term_df.get(term, 0)
        guarded[term] = tuple(keep)
    return ReplicatedTermPartition(
        guarded, base.node_strength, base.edges_df, base.clusters_df,
    )


def load_partition(source_dir: Path) -> TermPartition | ReplicatedTermPartition:
    """Load whichever partition kind was saved into ``source_dir``."""
    source_dir = Path(source_dir)
    if (source_dir / "term_to_shards.parquet").exists():
        return ReplicatedTermPartition.load(source_dir)
    return TermPartition.load(source_dir)
