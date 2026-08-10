"""NPMI-weighted term co-occurrence graph.

The graph vocabulary is intentionally narrower than the routing tokenizer:
``CountVectorizer`` below only accepts letter-only tokens of two or more
characters, so numbers and single-letter fragments never become graph
nodes.  Such terms still get a shard via the fallback
(:meth:`~sharded_index.partition.TermPartition.with_hash_fallback` or
:meth:`~sharded_index.partition.TermPartition.with_affinity_fallback`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .text import STOP_WORDS

EDGE_COLUMNS = ["src", "dst", "count", "weight"]
"""Columns of the edge-list DataFrame; ``weight`` is the NPMI value."""


def _llr(k11: np.ndarray, cx: np.ndarray, cy: np.ndarray, n: float) -> np.ndarray:
    """Dunning log-likelihood ratio for 2x2 co-occurrence contingencies."""
    k12, k21 = cx - k11, cy - k11
    k22 = n - cx - cy + k11

    def cell(k: np.ndarray, expected: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            value = k * np.log((k * n) / np.maximum(expected, 1e-12))
        return np.where(k > 0, value, 0.0)

    llr = 2.0 * (
        cell(k11, cx * cy) + cell(k12, cx * (n - cy))
        + cell(k21, (n - cx) * cy) + cell(k22, (n - cx) * (n - cy))
    )
    return np.maximum(llr, 0.0)


def _chi2(k11: np.ndarray, cx: np.ndarray, cy: np.ndarray, n: float) -> np.ndarray:
    """Pearson chi-squared for the same contingencies."""
    k12, k21 = cx - k11, cy - k11
    k22 = n - cx - cy + k11
    denom = np.maximum(cx * cy * (n - cx) * (n - cy), 1.0)
    return n * (k11 * k22 - k12 * k21) ** 2 / denom


def build_npmi_graph(
    texts: list[str],
    *,
    min_df: int = 2,
    max_df_ratio: float = 0.5,
    min_pair_count: int = 2,
    min_npmi: float = 0.0,
    edge_weight: str = "npmi",
    extra_stop_words: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build a term co-occurrence graph from texts.

    Two terms are connected if they co-occur in at least ``min_pair_count``
    texts and their NPMI passes ``min_npmi`` (the edge SET is always
    NPMI-filtered, so weighting schemes share one topology).  The edge
    weight is selected by ``edge_weight``: ``npmi`` (default, ∈ [-1, 1]),
    ``llr`` (Dunning log-likelihood ratio — punishes frequent co-occurrence
    harder, more stable on rare pairs) or ``chi2``.

    Parameters
    ----------
    texts:
        Corpus to build co-occurrence from (typically queries).
    min_df, max_df_ratio, min_pair_count, min_npmi:
        Filtering thresholds; defaults mirror :class:`~sharded_index.config.PartitionConfig`.
    edge_weight:
        ``npmi`` | ``llr`` | ``chi2``.
    extra_stop_words:
        Additional stop list applied only to the graph vocabulary (e.g.
        question-frame words); routing and indexing are unaffected, the
        global :data:`~sharded_index.text.STOP_WORDS` must stay untouched —
        document ids depend on it.

    Returns
    -------
    pd.DataFrame
        Edge list with columns :data:`EDGE_COLUMNS`, sorted by weight descending.
    """
    if edge_weight not in ("npmi", "llr", "chi2"):
        raise ValueError(f"unknown edge_weight: {edge_weight!r}")
    if not texts:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b[a-zа-яё]{2,}\b",
        stop_words=list(STOP_WORDS.union(extra_stop_words)),
        min_df=min_df,
        max_df=max_df_ratio,
        binary=True,
    )
    doc_term = vectorizer.fit_transform(texts)
    n_docs = float(doc_term.shape[0])
    vocab = vectorizer.get_feature_names_out()
    doc_freq = np.asarray(doc_term.sum(axis=0)).ravel().astype(np.float64)

    # Term-term co-occurrence counts, diagonal removed.
    cooc = (doc_term.T @ doc_term).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()

    if min_pair_count > 1:
        cooc = cooc.multiply(cooc >= min_pair_count).tocsr()
        cooc.eliminate_zeros()

    upper = sp.triu(cooc, k=1).tocoo()
    if upper.nnz == 0:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    rows, cols = upper.row, upper.col
    counts = upper.data.astype(np.float64)

    # NPMI = PMI / -log P(a, b)
    pmi = np.log((counts * n_docs) / np.maximum(doc_freq[rows] * doc_freq[cols], 1.0))
    npmi = pmi / np.maximum(-np.log(counts / n_docs), 1e-12)

    keep = npmi >= min_npmi
    rows, cols, counts, npmi = rows[keep], cols[keep], counts[keep], npmi[keep]

    if edge_weight == "npmi":
        weight = npmi
    elif edge_weight == "llr":
        weight = _llr(counts, doc_freq[rows], doc_freq[cols], n_docs)
    else:
        weight = _chi2(counts, doc_freq[rows], doc_freq[cols], n_docs)

    return pd.DataFrame({
        "src": vocab[rows],
        "dst": vocab[cols],
        "count": counts.astype(np.int64),
        "weight": weight.astype(np.float64),
    }).sort_values(["weight", "count"], ascending=[False, False], ignore_index=True)


def compute_node_strength(edges_df: pd.DataFrame) -> dict[str, float]:
    """Node strength: sum of NPMI edge weights incident to each term.

    Used to rank terms within a cluster and to order shards for
    early-termination search (:meth:`TermPartition.ranked_shards_for_query`).
    """
    if edges_df.empty:
        return {}

    incident = pd.concat([
        edges_df[["src", "weight"]].rename(columns={"src": "term", "weight": "w"}),
        edges_df[["dst", "weight"]].rename(columns={"dst": "term", "weight": "w"}),
    ], ignore_index=True)
    return incident.groupby("term")["w"].sum().to_dict()
