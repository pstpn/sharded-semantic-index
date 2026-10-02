"""Reference rankings: what the unsharded index returns for the evaluation queries."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from whoosh import index as whoosh_index
from whoosh import scoring
from whoosh.qparser import OrGroup, QueryParser

from sharded_index.config import Bm25Config
from sharded_index.storage import save_arrays
from sharded_index.text import normalize_text

logger = logging.getLogger(__name__)

RANKING_VERSION = 1
"""Bump when the ranking code changes: saved rankings become stale."""


class ReferenceIndex:
    """The unsharded index searched with BM25, OR over the query terms.

    Rankings are arrays of corpus document positions (the order of ``doc_ids``).
    """

    def __init__(self, index_dir: Path, doc_ids: Sequence[str], bm25: Bm25Config) -> None:
        index = whoosh_index.open_dir(index_dir)
        self._parser = QueryParser("text", schema=index.schema, group=OrGroup)
        self.searcher = index.searcher(weighting=scoring.BM25F(B=bm25.b, K1=bm25.k1))
        position = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        reader = self.searcher.reader()
        self._positions = np.full(reader.doc_count_all(), -1, dtype=np.int32)
        for docnum, stored in reader.iter_docs():
            self._positions[docnum] = position[stored["doc_id"]]

    def parse(self, query: str) -> Any:
        """The Whoosh query of a text query."""
        return self._parser.parse(normalize_text(query))

    def search(self, query: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Document positions and scores in ranking order (all matches if ``limit`` is None)."""
        hits = self.searcher.search(self.parse(query), limit=limit).top_n
        docnums = np.fromiter((docnum for _, docnum in hits), dtype=np.int64, count=len(hits))
        scores = np.fromiter((score for score, _ in hits), dtype=np.float64, count=len(hits))
        return self._positions[docnums], scores

    def close(self) -> None:
        self.searcher.close()


@dataclass
class Rankings:
    """Depth-limited reference rankings of one query sample."""

    queries: list[str]
    arrays: list[np.ndarray]
    truncated: np.ndarray

    @property
    def evaluated(self) -> np.ndarray:
        """Queries the index returns something for.

        A query none of whose terms occurs in the corpus has no shard to
        probe and takes no part in any metric.
        """
        return np.array([len(array) > 0 for array in self.arrays], dtype=bool)

    def save(self, path: Path, key: str) -> None:
        """Write the rankings; ``key`` identifies the index and settings they come from."""
        lengths = np.array([len(array) for array in self.arrays], dtype=np.int64)
        flat = np.concatenate(self.arrays) if self.arrays else np.array([], dtype=np.int32)
        save_arrays(
            path,
            queries=np.array(self.queries, dtype=str),
            flat=flat.astype(np.int32),
            offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            truncated=self.truncated.astype(bool),
            key=np.array(key),
        )

    @classmethod
    def load(cls, path: Path) -> Rankings:
        """Read rankings written by :meth:`save`."""
        with np.load(path) as data:
            offsets, flat = data["offsets"], data["flat"]
            arrays = [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
            return cls(data["queries"].tolist(), arrays, data["truncated"].copy())

    @staticmethod
    def key_of(path: Path) -> str:
        """The key a rankings file was saved under."""
        with np.load(path) as data:
            return str(data["key"])


def ranking_key(fingerprint: str, depth: int, bm25: Bm25Config) -> str:
    """Identity of a set of rankings: the index content, the depth and the scoring."""
    return f"v{RANKING_VERSION}|{fingerprint}|depth={depth}|k1={bm25.k1}|b={bm25.b}"


def update_rankings(
    directory: Path,
    samples: Mapping[str, list[str]],
    reference: ReferenceIndex,
    depth: int,
    key: str,
) -> None:
    """Bring ``directory`` to hold ``<sample>.npz`` rankings of exactly the given samples.

    Rankings depend on the index and the query only, so a query already
    saved in the directory under the same ``key`` is not searched again,
    whichever sample it was in.  Each sample is written as soon as it is ranked.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    known: dict[str, tuple[np.ndarray, bool]] = {}
    for path in sorted(directory.glob("*.npz")):
        if Rankings.key_of(path) != key:
            continue
        saved = Rankings.load(path)
        for query, array, cut in zip(
            saved.queries, saved.arrays, saved.truncated.tolist(), strict=True
        ):
            known[query] = (array, cut)

    for sample, queries in samples.items():
        missing = [query for query in dict.fromkeys(queries) if query not in known]
        logger.info("%s: %d queries, %d to rank", sample, len(queries), len(missing))
        for query in missing:
            ranking, _ = reference.search(query, limit=depth)
            known[query] = (ranking, len(ranking) == depth)
        arrays = [known[query][0] for query in queries]
        truncated = np.array([known[query][1] for query in queries], dtype=bool)
        Rankings(list(queries), arrays, truncated).save(directory / f"{sample}.npz", key)

    for path in directory.glob("*.npz"):
        if path.stem not in samples:
            path.unlink()
