"""Term co-occurrence graph of a query log, weighted by an association measure."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from sharded_index.config import GraphConfig

EDGE_COLUMNS = ["src", "dst", "count", "weight"]
GRAPH_TOKEN_PATTERN = r"(?u)\b[a-zа-яё]{2,}\b"  # noqa: RUF001
"""Graph terms are letter-only: numbers never become nodes."""


def dunning_llr(k11: np.ndarray, cx: np.ndarray, cy: np.ndarray, n: float) -> np.ndarray:
    """Dunning's log-likelihood ratio G² of 2x2 contingency tables.

    ``k11`` is the joint count, ``cx`` and ``cy`` the marginals, ``n`` the
    total.  Over- and under-representation both score high.
    """
    k12, k21 = cx - k11, cy - k11
    k22 = n - cx - cy + k11

    def cell(k: np.ndarray, expected: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            value = k * np.log((k * n) / np.maximum(expected, 1e-12))
        return np.where(k > 0, value, 0.0)

    llr = 2.0 * (
        cell(k11, cx * cy)
        + cell(k12, cx * (n - cy))
        + cell(k21, (n - cx) * cy)
        + cell(k22, (n - cx) * (n - cy))
    )
    return np.maximum(llr, 0.0)


def _chi2(k11: np.ndarray, cx: np.ndarray, cy: np.ndarray, n: float) -> np.ndarray:
    """Pearson's chi-squared of the same tables."""
    k12, k21 = cx - k11, cy - k11
    k22 = n - cx - cy + k11
    denominator = np.maximum(cx * cy * (n - cx) * (n - cy), 1.0)
    return n * (k11 * k22 - k12 * k21) ** 2 / denominator


def _unique_term_sets(text_terms: sp.csr_matrix) -> sp.csr_matrix:
    """Keep one row per distinct non-empty set of terms."""
    seen: set[tuple[int, ...]] = set()
    keep = []
    for row in range(text_terms.shape[0]):
        terms = tuple(text_terms.indices[text_terms.indptr[row] : text_terms.indptr[row + 1]])
        if terms and terms not in seen:
            seen.add(terms)
            keep.append(row)
    return text_terms[keep]


def build_graph(texts: list[str], config: GraphConfig, stop_words: Collection[str]) -> pd.DataFrame:
    """Build the edge list ``[src, dst, count, weight]`` from texts (queries).

    Texts with the same set of graph terms count once.  Two terms are linked
    if they share at least ``min_pair_count`` texts and their NPMI reaches
    ``min_npmi``; the topology is therefore the same for every
    ``edge_weight``.  Edges are sorted by weight, then count, descending.
    """
    if not texts:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    vectorizer = CountVectorizer(
        token_pattern=GRAPH_TOKEN_PATTERN,
        stop_words=sorted(stop_words),
        min_df=config.min_df,
        max_df=config.max_df_ratio,
        binary=True,
    )
    text_terms = vectorizer.fit_transform(texts).tocsr()
    text_terms.sort_indices()
    text_terms = _unique_term_sets(text_terms)
    n_texts = float(text_terms.shape[0])
    vocabulary = vectorizer.get_feature_names_out()
    frequency = np.asarray(text_terms.sum(axis=0)).ravel().astype(np.float64)

    cooccurrence = (text_terms.T @ text_terms).tocsr()
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    if config.min_pair_count > 1:
        cooccurrence = cooccurrence.multiply(cooccurrence >= config.min_pair_count).tocsr()
        cooccurrence.eliminate_zeros()

    upper = sp.triu(cooccurrence, k=1).tocoo()
    if upper.nnz == 0:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    rows, cols = upper.row, upper.col
    counts = upper.data.astype(np.float64)

    pmi = np.log((counts * n_texts) / np.maximum(frequency[rows] * frequency[cols], 1.0))
    npmi = pmi / np.maximum(-np.log(counts / n_texts), 1e-12)
    keep = npmi >= config.min_npmi
    rows, cols, counts, npmi = rows[keep], cols[keep], counts[keep], npmi[keep]

    if config.edge_weight == "npmi":
        weight = npmi
    elif config.edge_weight == "llr":
        weight = dunning_llr(counts, frequency[rows], frequency[cols], n_texts)
    else:
        weight = _chi2(counts, frequency[rows], frequency[cols], n_texts)

    edges = pd.DataFrame(
        {
            "src": vocabulary[rows],
            "dst": vocabulary[cols],
            "count": counts.astype(np.int64),
            "weight": weight.astype(np.float64),
        }
    )
    return edges.sort_values(["weight", "count"], ascending=[False, False], ignore_index=True)


def graph_terms(edges: pd.DataFrame) -> list[str]:
    """Graph nodes in first-appearance order (sources first, then targets)."""
    return pd.unique(pd.concat([edges["src"], edges["dst"]], ignore_index=True)).tolist()


def node_strength(edges: pd.DataFrame) -> dict[str, float]:
    """Sum of the weights of the edges incident to every term."""
    if edges.empty:
        return {}
    incident = pd.concat(
        [
            edges[["src", "weight"]].rename(columns={"src": "term"}),
            edges[["dst", "weight"]].rename(columns={"dst": "term"}),
        ],
        ignore_index=True,
    )
    return incident.groupby("term")["weight"].sum().to_dict()
