"""
Semantic Sharded Index — Leiden-clustering-based term partitioning for inverted index sharding.

Architecture
------------
1. Build NPMI-weighted term co-occurrence graph from a text corpus.
2. Partition terms into shards using Leiden community detection → term→shard mapping.
3. Assign each document to **ALL** shards where its terms appear.
4. Build a Whoosh BM25 index per shard (with stop-word filtering).
5. Search: route query to **ALL** shards containing its terms, merge results.

Key principle
-------------
Tokenization for routing **must** match tokenization at index time.
Stop words are filtered in ``normalize_text()`` / ``tokenize()``, in the
Whoosh analyzer (``StopFilter``), and in the ``CountVectorizer`` — all
using the same ``STOP_WORDS`` frozenset.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from whoosh import index as whoosh_index
from whoosh.analysis import LowercaseFilter, RegexTokenizer, StopFilter
from whoosh.fields import ID, TEXT, Schema
from whoosh.qparser import OrGroup, QueryParser

import igraph as ig
import leidenalg as la


# ═══════════════════════════════════════════════════════════════════════════
# Stop words & text processing
# ═══════════════════════════════════════════════════════════════════════════

STOP_WORDS: frozenset[str] = frozenset(ENGLISH_STOP_WORDS)
"""English stop words — filtered from queries, documents, NPMI graph, and Whoosh indices."""

WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Lowercase, extract alphanumeric tokens, remove stop words, re-join with spaces."""
    text = str(text).lower()
    tokens = WORD_RE.findall(text)
    return " ".join(t for t in tokens if t not in STOP_WORDS)


def tokenize(text: str) -> list[str]:
    """Tokenize text for routing — identical to ``normalize_text(text).split()``.

    This is the **only** tokenizer used for routing and document assignment.
    It must match the analyzer used in the Whoosh schema.
    Stop words are removed by ``normalize_text()``.
    """
    return normalize_text(text).split()


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ShardedIndexConfig:
    """Configuration for the semantic sharded index pipeline."""

    # NPMI graph parameters
    min_df: int = 2
    """Minimum document frequency for a term to be included in the NPMI graph."""

    max_df_ratio: float = 0.5
    """Maximum document frequency ratio. Terms appearing in more than this
    fraction of documents are excluded from the graph."""

    min_pair_count: int = 2
    """Minimum co-occurrence count for an edge to be kept."""

    min_npmi: float = 0.0
    """Minimum NPMI value for an edge to be kept."""

    # Leiden clustering parameters
    leiden_resolution: float = 1.0
    """Leiden resolution parameter. Higher → more, smaller clusters."""

    leiden_seed: int = 42
    """Random seed for Leiden clustering reproducibility."""

    # Index parameters
    index_root: Path = Path("data/sharded_whoosh")
    """Root directory for Whoosh index storage."""


# ═══════════════════════════════════════════════════════════════════════════
# NPMI graph construction
# ═══════════════════════════════════════════════════════════════════════════

def build_npmi_graph(
    texts: list[str],
    *,
    min_df: int = 5,
    max_df_ratio: float = 0.5,
    min_pair_count: int = 5,
    min_npmi: float = 0.0,
) -> pd.DataFrame:
    """Build NPMI-weighted term co-occurrence graph from texts.

    Uses ``CountVectorizer`` internally for computing co-occurrence counts.
    The vectorizer is **not** exposed — it is an implementation detail of
    the graph construction, not a tokenizer for routing.

    Parameters
    ----------
    texts : list[str]
        Texts to build co-occurrence from (typically queries or documents).
    min_df, max_df_ratio, min_pair_count, min_npmi :
        Filtering thresholds (see :class:`ShardedIndexConfig`).

    Returns
    -------
    pd.DataFrame
        Edge list with columns ``[src, dst, count, weight]`` where
        *weight* is the NPMI value.
    """
    if not texts:
        return pd.DataFrame(columns=["src", "dst", "count", "weight"])

    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b[a-zа-яё]+\b",
        stop_words=list(STOP_WORDS),
        min_df=min_df,
        max_df=max_df_ratio,
        binary=True,
    )
    X = vectorizer.fit_transform(texts)
    N = float(X.shape[0])
    vocab = vectorizer.get_feature_names_out()
    df = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)

    # Co-occurrence matrix (upper triangle only)
    C = (X.T @ X).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()

    if min_pair_count > 1:
        C = C.multiply(C >= min_pair_count).tocsr()
        C.eliminate_zeros()

    Cu = sp.triu(C, k=1).tocoo()
    if Cu.nnz == 0:
        return pd.DataFrame(columns=["src", "dst", "count", "weight"])

    rows = Cu.row
    cols = Cu.col
    counts = Cu.data.astype(np.float64)

    # PMI → NPMI
    denom = np.maximum(df[rows] * df[cols], 1.0)
    pmi = np.log((counts * N) / denom)
    norm = np.maximum(-np.log(counts / N), 1e-12)
    npmi = pmi / norm

    # Filter by minimum NPMI
    keep = npmi >= min_npmi
    rows, cols, counts, npmi = rows[keep], cols[keep], counts[keep], npmi[keep]

    return pd.DataFrame({
        "src": vocab[rows],
        "dst": vocab[cols],
        "count": counts.astype(np.int64),
        "weight": npmi.astype(np.float64),
    }).sort_values(["weight", "count"], ascending=[False, False], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# Leiden clustering
# ═══════════════════════════════════════════════════════════════════════════

def cluster_terms(
    edges_df: pd.DataFrame,
    *,
    resolution: float = 1.0,
    seed: int = 42,
    weight_col: str = "weight",
) -> tuple[dict[str, int], pd.DataFrame]:
    """Partition terms into clusters using Leiden community detection.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Edge list with columns ``[src, dst, weight]``.
    resolution : float
        Leiden resolution parameter. Higher → more, smaller clusters.
    seed : int
        Random seed for reproducibility.
    weight_col : str
        Column name for edge weights.

    Returns
    -------
    term_to_shard : dict[str, int]
        Mapping from term to shard (cluster) ID.
    clusters_df : pd.DataFrame
        DataFrame with columns ``[cluster, size]`` sorted by size descending.
    """
    if edges_df.empty:
        return {}, pd.DataFrame(columns=["cluster", "size"])

    nodes = pd.Index(pd.unique(
        pd.concat([edges_df["src"], edges_df["dst"]], ignore_index=True)
    ))
    node_to_id = {n: i for i, n in enumerate(nodes)}

    src_ids = edges_df["src"].map(node_to_id).astype(np.int32).to_numpy()
    dst_ids = edges_df["dst"].map(node_to_id).astype(np.int32).to_numpy()
    weights = np.maximum(edges_df[weight_col].astype(float).to_numpy(), 0.0)

    g = ig.Graph(n=len(nodes), edges=list(zip(src_ids, dst_ids)), directed=False)
    g.es["weight"] = weights.tolist()

    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=int(seed),
    )

    membership = np.asarray(part.membership, dtype=np.int32)
    term_to_shard = {str(nodes[i]): int(membership[i]) for i in range(len(nodes))}
    sizes = pd.Series(membership).value_counts().sort_values(ascending=False)
    clusters_df = pd.DataFrame({
        "cluster": sizes.index.astype(int),
        "size": sizes.values.astype(int),
    })

    return term_to_shard, clusters_df


# ═══════════════════════════════════════════════════════════════════════════
# Node strength (sum of NPMI edge weights per term)
# ═══════════════════════════════════════════════════════════════════════════

def compute_node_strength(
    edges_df: pd.DataFrame,
    weight_col: str = "weight",
) -> dict[str, float]:
    """Compute node strength: sum of edge weights for each term.

    Useful for ranking terms within a shard or for weighted shard ordering.
    """
    if edges_df.empty:
        return {}

    strengths = pd.concat([
        edges_df[["src", weight_col]].rename(columns={"src": "term", weight_col: "w"}),
        edges_df[["dst", weight_col]].rename(columns={"dst": "term", weight_col: "w"}),
    ], ignore_index=True)
    return strengths.groupby("term")["w"].sum().to_dict()


# ═══════════════════════════════════════════════════════════════════════════
# ShardedIndex — term-to-shard partition
# ═══════════════════════════════════════════════════════════════════════════

class ShardedIndex:
    """Term-to-shard partition based on Leiden clustering of NPMI co-occurrence graph.

    Routes queries and documents to **ALL** shards containing their terms
    (not majority-vote single shard).

    Usage::

        index = ShardedIndex.from_corpus(texts, config=config)
        shards = index.get_shards_for_query("how to reset windows password")
        # shards = {3, 8, 5} — ALL shards for ALL query terms
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
        self.edges_df = (
            edges_df
            if edges_df is not None
            else pd.DataFrame(columns=["src", "dst", "count", "weight"])
        )
        self.clusters_df = (
            clusters_df
            if clusters_df is not None
            else pd.DataFrame(columns=["cluster", "size"])
        )
        self._n_shards = (
            max(term_to_shard.values(), default=-1) + 1
            if term_to_shard
            else 0
        )

    # ── Construction ──────────────────────────────────────────────────

    @classmethod
    def from_corpus(
        cls,
        texts: list[str],
        *,
        config: ShardedIndexConfig | None = None,
    ) -> ShardedIndex:
        """Build a :class:`ShardedIndex` from a text corpus.

        Steps:
        1. Build NPMI co-occurrence graph
        2. Cluster terms with Leiden
        3. Compute node strengths

        Parameters
        ----------
        texts : list[str]
            Texts to build the co-occurrence graph from.
        config : ShardedIndexConfig, optional
            Configuration. Uses defaults if not provided.
        """
        if config is None:
            config = ShardedIndexConfig()

        edges_df = build_npmi_graph(
            texts,
            min_df=config.min_df,
            max_df_ratio=config.max_df_ratio,
            min_pair_count=config.min_pair_count,
            min_npmi=config.min_npmi,
        )
        term_to_shard, clusters_df = cluster_terms(
            edges_df,
            resolution=config.leiden_resolution,
            seed=config.leiden_seed,
        )
        node_strength = compute_node_strength(edges_df)
        return cls(term_to_shard, node_strength, edges_df=edges_df, clusters_df=clusters_df)

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def n_shards(self) -> int:
        """Number of shards (clusters)."""
        return self._n_shards

    @property
    def n_terms(self) -> int:
        """Number of terms with a shard assignment."""
        return len(self.term_to_shard)

    # ── Routing ────────────────────────────────────────────────────────

    def get_shards_for_tokens(self, tokens: list[str]) -> set[int]:
        """Get **ALL** shards containing any of the given tokens."""
        return {
            self.term_to_shard[t]
            for t in tokens
            if t in self.term_to_shard
        }

    def get_shards_for_query(self, query: str) -> set[int]:
        """Get **ALL** shards for **ALL** terms in the query."""
        return self.get_shards_for_tokens(tokenize(query))

    def get_shards_for_document(self, doc_text: str) -> set[int]:
        """Get **ALL** shards for **ALL** terms in the document text."""
        return self.get_shards_for_tokens(tokenize(doc_text))

    def get_shards_for_query_ranked(self, query: str) -> list[int]:
        """Get shards for query, ranked by total node strength of matching terms.

        Shards with higher aggregate strength of the query terms they contain
        are returned first. Useful for early-termination strategies.
        """
        shards = self.get_shards_for_query(query)
        if not shards:
            return []

        # Compute strength per shard (sum of strengths of query terms in that shard)
        query_tokens = tokenize(query)
        shard_strength: dict[int, float] = defaultdict(float)
        for t in query_tokens:
            if t in self.term_to_shard and t in self.node_strength:
                shard_strength[self.term_to_shard[t]] += self.node_strength[t]

        return sorted(shards, key=lambda s: shard_strength.get(s, 0.0), reverse=True)

    # ── Statistics ─────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the partition."""
        if not self.term_to_shard:
            return {"n_shards": 0, "n_terms": 0}

        shard_sizes = defaultdict(int)
        for shard_id in self.term_to_shard.values():
            shard_sizes[shard_id] += 1

        sizes = list(shard_sizes.values())
        return {
            "n_shards": self._n_shards,
            "n_terms": len(self.term_to_shard),
            "shard_size_min": min(sizes),
            "shard_size_max": max(sizes),
            "shard_size_mean": float(np.mean(sizes)),
            "shard_size_median": float(np.median(sizes)),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Document assignment
# ═══════════════════════════════════════════════════════════════════════════

def assign_docs_to_shards(
    docs: dict[str, str],
    partitioner: ShardedIndex,
) -> tuple[dict[str, set[int]], dict[int, dict[str, str]]]:
    """Assign each document to **ALL** shards where its terms appear.

    Parameters
    ----------
    docs : dict[str, str]
        Mapping from doc_id to document text.
    partitioner : ShardedIndex
        Term-to-shard partition.

    Returns
    -------
    doc_to_shards : dict[str, set[int]]
        Mapping from doc_id to set of shard IDs.
    docs_by_shard : dict[int, dict[str, str]]
        Mapping from shard_id to ``{doc_id: doc_text}``.
    """
    doc_to_shards: dict[str, set[int]] = {}
    docs_by_shard: dict[int, dict[str, str]] = defaultdict(dict)

    for doc_id, doc_text in docs.items():
        shards = partitioner.get_shards_for_document(doc_text)
        doc_to_shards[doc_id] = shards
        for shard_id in shards:
            docs_by_shard[shard_id][doc_id] = doc_text

    return doc_to_shards, dict(docs_by_shard)


# ═══════════════════════════════════════════════════════════════════════════
# Whoosh index building
# ═══════════════════════════════════════════════════════════════════════════

# Whoosh analyzer: RegexTokenizer + LowercaseFilter + StopFilter
# Equivalent to normalize_text() / tokenize() — consistent stop-word removal.
WHOOSH_ANALYZER = RegexTokenizer() | LowercaseFilter() | StopFilter(stoplist=list(STOP_WORDS))

WHOOSH_SCHEMA = Schema(
    doc_id=ID(stored=True, unique=True),
    text=TEXT(stored=True, analyzer=WHOOSH_ANALYZER),
)
"""Whoosh schema with stop-word filtering.

The analyzer (``RegexTokenizer → LowercaseFilter → StopFilter``) is equivalent
to ``normalize_text(text).split()`` — both lowercase and remove stop words.
"""


def _build_single_shard(shard_id: int, shard_docs: dict[str, str], shard_dir: Path) -> int:
    """Build one Whoosh index shard. Called by worker threads/processes."""
    shard_dir.mkdir(parents=True, exist_ok=True)

    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        text=TEXT(stored=True, analyzer=WHOOSH_ANALYZER),
    )
    ix = whoosh_index.create_in(shard_dir, schema)
    writer = ix.writer(limitmb=256)
    for doc_id, text in shard_docs.items():
        writer.add_document(doc_id=doc_id, text=text)
    writer.commit(merge=False)
    return shard_id


def build_whoosh_indices(
    docs_by_shard: dict[int, dict[str, str]],
    index_root: Path,
    *,
    n_workers: int | None = None,
) -> None:
    """Build a Whoosh BM25 index for each shard in parallel.

    Uses ``ThreadPoolExecutor`` — Whoosh is I/O-bound (disk writes),
    so threads work well and avoid the pickle overhead of processes.

    Parameters
    ----------
    docs_by_shard : dict[int, dict[str, str]]
        Mapping from shard_id to ``{doc_id: doc_text}``.
    index_root : Path
        Root directory for index storage. Each shard is stored in
        ``index_root / "shard_{sid:03d}"``.
    n_workers : int, optional
        Number of parallel threads. Defaults to ``min(len(shards), os.cpu_count())``.
    """
    import concurrent.futures

    if index_root.exists():
        shutil.rmtree(index_root)
    index_root.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = min(len(docs_by_shard), os.cpu_count() or 1)

    shards = list(docs_by_shard.items())
    total = len(shards)

    if n_workers <= 1 or total <= 1:
        for i, (shard_id, shard_docs) in enumerate(shards):
            shard_dir = index_root / f"shard_{shard_id:03d}"
            _build_single_shard(shard_id, shard_docs, shard_dir)
            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"  [{i+1}/{total}] shards built")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for shard_id, shard_docs in shards:
                shard_dir = index_root / f"shard_{shard_id:03d}"
                futures[pool.submit(_build_single_shard, shard_id, shard_docs, shard_dir)] = shard_id

            done = 0
            for future in concurrent.futures.as_completed(futures):
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  [{done}/{total}] shards built")
                future.result()  # raise any exception


# ═══════════════════════════════════════════════════════════════════════════
# Search
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """A single search result from a sharded index."""

    doc_id: str
    score: float
    text: str
    shard_id: int


class ShardedSearcher:
    """Search across sharded Whoosh indices.

    Routes the query to **ALL** shards containing its terms,
    searches each shard, and merges results sorted by BM25 score.

    Whoosh indices are cached on first access via :meth:`_open_index`
    to avoid repeated disk I/O.
    """

    def __init__(self, index_root: Path, partitioner: ShardedIndex):
        self.index_root = Path(index_root)
        self.partitioner = partitioner
        self._ix_cache: dict[int, whoosh_index.Index] = {}

    def _open_index(self, shard_id: int) -> whoosh_index.Index | None:
        """Open a Whoosh index, caching the result for reuse."""
        if shard_id in self._ix_cache:
            return self._ix_cache[shard_id]
        shard_dir = self.index_root / f"shard_{shard_id:03d}"
        if not shard_dir.exists():
            return None
        ix = whoosh_index.open_dir(shard_dir)
        self._ix_cache[shard_id] = ix
        return ix

    def search_shard(
        self,
        query: str,
        shard_id: int,
        *,
        top_k: int | None = 10,
    ) -> list[SearchResult]:
        """Search a single shard.

        Parameters
        ----------
        query : str
            Search query text.
        shard_id : int
            Shard to search.
        top_k : int | None
            Maximum number of results. ``None`` = all matching documents.

        Returns
        -------
        list[SearchResult]
        """
        ix = self._open_index(shard_id)
        if ix is None:
            return []

        parser = QueryParser("text", schema=ix.schema, group=OrGroup)
        q = parser.parse(normalize_text(query))

        results: list[SearchResult] = []
        with ix.searcher() as searcher:
            hits = searcher.search(q, limit=top_k)
            for h in hits:
                results.append(SearchResult(
                    doc_id=h["doc_id"],
                    score=float(h.score),
                    text=h["text"],
                    shard_id=shard_id,
                ))
        return results

    def search(
        self,
        query: str,
        *,
        top_k: int | None = 10,
    ) -> list[SearchResult]:
        """Search **ALL** shards for **ALL** query terms, merge and sort results.

        Parameters
        ----------
        query : str
            Search query text.
        top_k : int | None
            Maximum number of results **per shard**. ``None`` = all results.

        Returns
        -------
        list[SearchResult]
            Merged results from all relevant shards, sorted by score descending.
        """
        shards = self.partitioner.get_shards_for_query(query)
        all_results: list[SearchResult] = []

        for shard_id in shards:
            all_results.extend(
                self.search_shard(query, shard_id, top_k=top_k)
            )

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results


# ═══════════════════════════════════════════════════════════════════════════
# MS MARCO v2.1 data loading
# ═══════════════════════════════════════════════════════════════════════════

def _iter_passages(row: dict[str, Any]) -> list[str]:
    """Extract passage texts from an MS MARCO row."""
    passages = row.get("passages", {}) or {}
    texts = passages.get("passage_text", []) or []
    selected = passages.get("is_selected", []) or []

    if not texts:
        return []

    # If selection flags exist, prefer selected passages
    if selected:
        selected_texts = [
            t for t, s in zip(texts, selected)
            if int(s) == 1 and str(t).strip()
        ]
        if selected_texts:
            return selected_texts

    # Fallback: all non-empty passages
    return [str(t) for t in texts if str(t).strip()]


def extract_query_doc_pairs(
    split,
    *,
    max_rows: int = 200_000,
    max_passages_per_query: int = 3,
    selected_only: bool = True,
) -> list[dict[str, str]]:
    """Extract query-document pairs from MS MARCO v2.1.

    Parameters
    ----------
    split : Dataset
        HuggingFace dataset split (e.g., ``ds["train"]``).
    max_rows : int
        Maximum number of rows to process.
    max_passages_per_query : int
        Maximum passages per query.
    selected_only : bool
        If True, only include passages marked as selected.

    Returns
    -------
    list[dict[str, str]]
        List of dicts with keys: ``query``, ``doc_id``, ``doc_text``.
    """
    pairs: list[dict[str, str]] = []

    n = min(len(split), max_rows)
    for i in range(n):
        row = split[i]
        query = normalize_text(row.get("query", ""))
        if not query:
            continue

        passages = _iter_passages(row)

        # If selected_only and the row has selection flags, re-filter
        if selected_only:
            sel_flags = (
                row.get("passages", {}).get("is_selected", [])
                or []
            )
            if sel_flags:
                passage_texts = (
                    row.get("passages", {}).get("passage_text", [])
                    or []
                )
                passages = [
                    p for p, f in zip(passage_texts, sel_flags)
                    if int(f) == 1
                ]

        if not passages:
            continue

        for passage_text in passages[:max_passages_per_query]:
            doc_text = normalize_text(passage_text)
            if not doc_text:
                continue

            doc_id = hashlib.md5(doc_text.encode("utf-8")).hexdigest()
            pairs.append({
                "query": query,
                "doc_id": doc_id,
                "doc_text": doc_text,
            })

    return pairs
