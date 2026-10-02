"""BM25 with collection-wide statistics for searching a shard.

A shard stores whole documents, so term frequencies and document lengths are
the same in every shard.  Scored with the document count, document
frequencies and average length of the whole collection, a shard gives each
document exactly the score of the unsharded index.
"""

from __future__ import annotations

from math import log
from typing import Any

from whoosh import scoring


class GlobalBM25(scoring.BM25F):
    """Whoosh BM25F that takes collection statistics from a reference searcher."""

    def __init__(self, reference: Any, b: float, k1: float) -> None:
        super().__init__(B=b, K1=k1)
        self.reference = reference

    def scorer(self, searcher: Any, fieldname: str, text: bytes, qf: int = 1) -> scoring.BaseScorer:
        return _GlobalBM25Scorer(searcher, self.reference, fieldname, text, self.B, self.K1, qf)


class _GlobalBM25Scorer(scoring.WeightLengthScorer):
    def __init__(
        self,
        searcher: Any,
        reference: Any,
        fieldname: str,
        text: bytes,
        b: float,
        k1: float,
        qf: int = 1,
    ) -> None:
        frequency = reference.doc_frequency(fieldname, text)
        self.idf = log(reference.doc_count_all() / (frequency + 1)) + 1
        self.avgfl = reference.avg_field_length(fieldname) or 1
        self.B, self.K1, self.qf = b, k1, qf
        self.setup(searcher, fieldname, text)

    def _score(self, weight: float, length: float) -> float:
        return scoring.bm25(self.idf, weight, length, self.avgfl, self.B, self.K1)
