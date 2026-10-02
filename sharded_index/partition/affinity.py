"""Document vote: out-of-graph terms join the cluster they co-occur with in documents."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import scipy.sparse as sp

from sharded_index.data.corpus import Corpus
from sharded_index.graph.cooccurrence import dunning_llr
from sharded_index.partition.hashing import hash_shard, id_space


def significant_choice(votes: sp.spmatrix, mass: np.ndarray, method: str) -> np.ndarray:
    """Pick a group for every row of a terms-by-groups vote matrix.

    ``votes[t, g]`` counts (document, graph term of group ``g``) pairs over
    the documents holding ``t``; ``mass[g]`` is the same count over the whole
    corpus.  ``raw`` picks the group with the most votes, which favours groups
    of ubiquitous terms.  ``llr`` picks the group in which the term is most
    significantly over-represented (Dunning's G², negated for
    under-representation).  Ties go to the smaller group id; a row without
    votes gets ``-1``.
    """
    votes = sp.csr_matrix(votes)
    votes.sum_duplicates()
    votes.sort_indices()
    choice = np.full(votes.shape[0], -1, dtype=np.int64)
    if votes.nnz == 0:
        return choice

    rows = np.repeat(np.arange(votes.shape[0]), np.diff(votes.indptr))
    counts = votes.data.astype(np.float64)
    if method == "raw":
        score = counts
    elif method == "llr":
        mass = np.asarray(mass, dtype=np.float64)
        total = float(mass.sum())
        row_totals = np.asarray(votes.sum(axis=1)).ravel()[rows].astype(np.float64)
        col_totals = mass[votes.indices]
        g2 = dunning_llr(counts, row_totals, col_totals, total)
        score = np.where(counts * total >= row_totals * col_totals, g2, -g2)
    else:
        msg = f"unknown vote method: {method!r}"
        raise ValueError(msg)

    order = np.lexsort((votes.indices, -score, rows))
    first = order[np.r_[True, rows[order][1:] != rows[order][:-1]]]
    choice[rows[first]] = votes.indices[first]
    return choice


def with_affinity_fallback(
    clustering: Mapping[str, int], corpus: Corpus, vote: str
) -> dict[str, int]:
    """Extend a clustering to the corpus vocabulary by document co-occurrence.

    Every document of an out-of-graph term votes with the clusters of its
    graph terms, one vote per graph term; :func:`significant_choice` picks
    the cluster.  Only terms that never share a document with a graph term
    are hashed into the separate id space ``[n, 2n)``.
    """
    n = id_space(clustering)
    vocabulary = corpus.vocabulary
    cluster = np.array([clustering.get(term, -1) for term in vocabulary], dtype=np.int64)
    known = cluster >= 0
    term_cluster = sp.csr_matrix(
        (np.ones(int(known.sum())), (np.flatnonzero(known), cluster[known])),
        shape=(len(vocabulary), max(n, 1)),
    )
    doc_clusters = (corpus.by_term @ term_cluster).tocsr()
    mass = np.asarray(doc_clusters.sum(axis=0)).ravel()
    tail = np.flatnonzero(~known)
    cluster[tail] = significant_choice(corpus.by_term[:, tail].T @ doc_clusters, mass, vote)

    return {
        term: int(cluster[j]) if cluster[j] >= 0 else n + hash_shard(term, max(n, 1))
        for j, term in enumerate(vocabulary)
    }
