"""The term partition: which shards hold each term."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

PARTITION_COLUMNS = ["term", "shard", "rank"]


@dataclass(frozen=True)
class TermPartition:
    """Term → shards.  The first shard of a term is its primary one; the rest are replicas.

    A document lives in every shard of each of its terms, and a query can be
    answered from any set of shards that holds a shard of each of its terms.
    """

    term_shards: dict[str, tuple[int, ...]]

    @classmethod
    def from_primary(cls, term_to_shard: Mapping[str, int]) -> TermPartition:
        """A partition without replicas."""
        return cls({term: (int(shard),) for term, shard in term_to_shard.items()})

    @property
    def primary(self) -> dict[str, int]:
        """Term → its primary shard."""
        return {term: shards[0] for term, shards in self.term_shards.items()}

    @property
    def n_terms(self) -> int:
        return len(self.term_shards)

    @property
    def n_shards(self) -> int:
        """Size of the shard id space (largest id + 1); some ids may be unused."""
        return max((max(shards) for shards in self.term_shards.values()), default=-1) + 1

    @property
    def replication_factor(self) -> float:
        """Mean number of shards per term."""
        if not self.term_shards:
            return 0.0
        return sum(len(shards) for shards in self.term_shards.values()) / len(self.term_shards)

    def shards_of(self, terms: Iterable[str]) -> set[int]:
        """Every shard holding any of the terms; unknown terms are ignored."""
        return {shard for term in terms for shard in self.term_shards.get(term, ())}

    def incidence(self, vocabulary: Sequence[str]) -> sp.csr_matrix:
        """Binary vocabulary-by-shards matrix of the partition."""
        rows, cols = [], []
        for row, term in enumerate(vocabulary):
            for shard in self.term_shards.get(term, ()):
                rows.append(row)
                cols.append(shard)
        return sp.csr_matrix(
            (np.ones(len(rows), dtype=np.int32), (rows, cols)),
            shape=(len(vocabulary), self.n_shards),
        )

    def save(self, path: Path) -> None:
        """Write the partition as a ``[term, shard, rank]`` table."""
        rows = [
            (term, shard, rank)
            for term, shards in self.term_shards.items()
            for rank, shard in enumerate(shards)
        ]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=PARTITION_COLUMNS).to_parquet(path, index=False)

    @classmethod
    def load(cls, path: Path) -> TermPartition:
        """Read a partition written by :meth:`save`."""
        table = pd.read_parquet(path).sort_values(["term", "rank"], kind="stable")
        term_shards: dict[str, list[int]] = {}
        for term, shard in zip(table["term"].tolist(), table["shard"].tolist(), strict=True):
            term_shards.setdefault(term, []).append(shard)
        return cls({term: tuple(shards) for term, shards in term_shards.items()})
