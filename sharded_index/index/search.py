"""Search over physical shard indices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from whoosh import index as whoosh_index

from sharded_index.index.bm25 import GlobalBM25
from sharded_index.index.build import shard_path


class ShardSearcher:
    """Searches shard indices under one root with collection-wide BM25 statistics."""

    def __init__(self, root: Path, weighting: GlobalBM25) -> None:
        self.root = Path(root)
        self.weighting = weighting
        self._indices: dict[int, Any] = {}

    def scores(self, shard_id: int, parsed_query: Any) -> dict[str, float]:
        """``doc_id → score`` of every document of the shard matching the query."""
        if shard_id not in self._indices:
            self._indices[shard_id] = whoosh_index.open_dir(shard_path(self.root, shard_id))
        with self._indices[shard_id].searcher(weighting=self.weighting) as searcher:
            return {
                hit["doc_id"]: float(hit.score) for hit in searcher.search(parsed_query, limit=None)
            }
