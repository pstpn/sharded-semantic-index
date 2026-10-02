"""The tokenized corpus: which vocabulary terms occur in which documents."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sharded_index.storage import load_incidence, save_incidence
from sharded_index.text import Tokenizer


@dataclass(frozen=True)
class Corpus:
    """Binary documents-by-vocabulary incidence with a sorted vocabulary."""

    doc_ids: list[str]
    vocabulary: list[str]
    doc_terms: sp.csr_matrix

    @classmethod
    def from_documents(cls, documents: Mapping[str, str], tokenizer: Tokenizer) -> Corpus:
        """Tokenize ``doc_id → text`` documents."""
        token_sets = [set(tokenizer(text)) for text in documents.values()]
        return cls.from_token_sets(list(documents), token_sets)

    @classmethod
    def from_token_sets(
        cls, doc_ids: Sequence[str], token_sets: Sequence[Collection[str]]
    ) -> Corpus:
        """Build the incidence from the set of terms of every document."""
        vocabulary = sorted(set().union(*token_sets)) if token_sets else []
        column = {term: j for j, term in enumerate(vocabulary)}
        indptr, indices = [0], []
        for tokens in token_sets:
            indices.extend(sorted(column[term] for term in set(tokens)))
            indptr.append(len(indices))
        matrix = sp.csr_matrix(
            (np.ones(len(indices), dtype=np.int32), indices, indptr),
            shape=(len(token_sets), len(vocabulary)),
        )
        return cls(list(doc_ids), vocabulary, matrix)

    @property
    def n_docs(self) -> int:
        return int(self.doc_terms.shape[0])

    @cached_property
    def df(self) -> np.ndarray:
        """Document frequency of every vocabulary term."""
        return np.asarray(self.doc_terms.sum(axis=0)).ravel().astype(np.int64)

    @cached_property
    def by_term(self) -> sp.csc_matrix:
        """The incidence in column-oriented form (documents of a term)."""
        return self.doc_terms.tocsc()

    def save(self, directory: Path) -> None:
        """Write the corpus to ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_incidence(directory / "doc_terms.npz", self.doc_terms)
        pd.DataFrame({"term": self.vocabulary, "df": self.df}).to_parquet(
            directory / "vocabulary.parquet", index=False
        )
        pd.DataFrame({"doc_id": self.doc_ids}).to_parquet(
            directory / "documents.parquet", index=False
        )

    @classmethod
    def load(cls, directory: Path) -> Corpus:
        """Read a corpus written by :meth:`save`."""
        directory = Path(directory)
        return cls(
            pd.read_parquet(directory / "documents.parquet")["doc_id"].tolist(),
            pd.read_parquet(directory / "vocabulary.parquet")["term"].tolist(),
            load_incidence(directory / "doc_terms.npz", dtype=np.int32),
        )
