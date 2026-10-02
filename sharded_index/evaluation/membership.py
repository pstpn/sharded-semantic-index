"""Document → shard membership implied by a term partition."""

from __future__ import annotations

from collections.abc import Collection, Iterable

import numpy as np
import scipy.sparse as sp

from sharded_index.data.corpus import Corpus
from sharded_index.partition.model import TermPartition


def assign_documents(partition: TermPartition, corpus: Corpus) -> sp.csr_matrix:
    """Boolean documents-by-shards matrix: a document lives in every shard of its terms."""
    return sp.csr_matrix((corpus.doc_terms @ partition.incidence(corpus.vocabulary)) > 0)


def assign_by_lookup(
    partition: TermPartition, token_sets: Iterable[Collection[str]]
) -> sp.csr_matrix:
    """The same membership computed document by document, without the corpus matrix.

    An independent path used to verify :func:`assign_documents`.
    """
    indptr, indices = [0], []
    for tokens in token_sets:
        indices.extend(sorted(partition.shards_of(tokens)))
        indptr.append(len(indices))
    return sp.csr_matrix(
        (np.ones(len(indices), dtype=bool), indices, indptr),
        shape=(len(indptr) - 1, partition.n_shards),
    )


def same_membership(a: sp.spmatrix, b: sp.spmatrix) -> bool:
    """Whether two membership matrices hold the same (document, shard) pairs."""
    if a.shape != b.shape:
        return False
    return (sp.csr_matrix(a, dtype=bool) != sp.csr_matrix(b, dtype=bool)).nnz == 0


def shard_volumes(membership: sp.spmatrix) -> np.ndarray:
    """Number of documents in every shard."""
    return np.asarray(membership.sum(axis=0)).ravel()
