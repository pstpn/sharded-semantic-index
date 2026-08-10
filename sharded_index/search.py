"""Search across sharded Whoosh indices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whoosh import index as whoosh_index
from whoosh.qparser import OrGroup, QueryParser

from .indexing import shard_dir
from .partition import TermPartition
from .text import normalize_text


@dataclass
class SearchResult:
    """A single hit from a sharded search."""

    doc_id: str
    score: float
    text: str
    shard_id: int


class ShardedSearcher:
    """Fan a query out to **all** shards holding its terms and merge results.

    Opened Whoosh indices are cached, so repeated queries touch the disk only
    on the first access to each shard.
    """

    def __init__(self, index_root: Path, partition: TermPartition):
        self.index_root = Path(index_root)
        self.partition = partition
        self._index_cache: dict[int, whoosh_index.Index] = {}

    def _open_index(self, shard_id: int) -> whoosh_index.Index | None:
        if shard_id in self._index_cache:
            return self._index_cache[shard_id]
        target_dir = shard_dir(self.index_root, shard_id)
        if not target_dir.exists():
            return None
        ix = whoosh_index.open_dir(target_dir)
        self._index_cache[shard_id] = ix
        return ix

    def search_shard(
        self,
        query: str,
        shard_id: int,
        *,
        top_k: int | None = 10,
    ) -> list[SearchResult]:
        """BM25 search in a single shard (OR over query terms).

        ``top_k=None`` returns all matching documents.
        """
        ix = self._open_index(shard_id)
        if ix is None:
            return []

        parser = QueryParser("text", schema=ix.schema, group=OrGroup)
        parsed = parser.parse(normalize_text(query))

        with ix.searcher() as searcher:
            return [
                SearchResult(
                    doc_id=hit["doc_id"],
                    score=float(hit.score),
                    text=hit["text"],
                    shard_id=shard_id,
                )
                for hit in searcher.search(parsed, limit=top_k)
            ]

    def search(
        self,
        query: str,
        *,
        top_k: int | None = 10,
    ) -> list[SearchResult]:
        """Search all shards relevant to the query, merge by BM25 score.

        ``top_k`` limits results **per shard**; the merged list is sorted by
        score descending.
        """
        results: list[SearchResult] = []
        for shard_id in self.partition.shards_for_query(query):
            results.extend(self.search_shard(query, shard_id, top_k=top_k))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
