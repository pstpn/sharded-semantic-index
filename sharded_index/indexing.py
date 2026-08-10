"""Whoosh BM25 index per shard.

The analyzer (``RegexTokenizer → LowercaseFilter → StopFilter``) matches
:func:`~sharded_index.text.tokenize`: both lowercase and drop the same stop
words, so routing sees exactly the terms Whoosh indexed.
"""

from __future__ import annotations

import concurrent.futures
import os
import shutil
from pathlib import Path

from whoosh import index as whoosh_index
from whoosh.analysis import LowercaseFilter, RegexTokenizer, StopFilter
from whoosh.fields import ID, TEXT, Schema

from .text import STOP_WORDS

WHOOSH_ANALYZER = RegexTokenizer() | LowercaseFilter() | StopFilter(stoplist=list(STOP_WORDS))

WHOOSH_SCHEMA = Schema(
    doc_id=ID(stored=True, unique=True),
    text=TEXT(stored=True, analyzer=WHOOSH_ANALYZER),
)


def shard_dir(index_root: Path, shard_id: int) -> Path:
    """Directory of one shard's index: ``index_root/shard_042``."""
    return Path(index_root) / f"shard_{shard_id:03d}"


def _build_shard(shard_docs: dict[str, str], target_dir: Path) -> None:
    """Build a single Whoosh index (executed by worker threads)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    ix = whoosh_index.create_in(target_dir, WHOOSH_SCHEMA)
    writer = ix.writer(limitmb=256)
    for doc_id, text in shard_docs.items():
        writer.add_document(doc_id=doc_id, text=text)
    writer.commit(merge=False)


def build_whoosh_indices(
    docs_by_shard: dict[int, dict[str, str]],
    index_root: Path,
    *,
    n_workers: int | None = None,
    overwrite: bool = False,
) -> None:
    """Build a Whoosh BM25 index for every shard, in parallel.

    Whoosh is I/O-bound, so a ``ThreadPoolExecutor`` parallelizes well without
    the pickling overhead of processes.

    Parameters
    ----------
    docs_by_shard:
        ``shard_id → {doc_id: doc_text}`` (see :func:`assign_docs_to_shards`).
    index_root:
        Root directory; each shard goes to ``index_root/shard_{id:03d}``.
    n_workers:
        Thread count, default ``min(n_shards, cpu_count)``.
    overwrite:
        If the root already exists: ``True`` rebuilds from scratch,
        ``False`` (default) keeps it and skips the build entirely.
    """
    index_root = Path(index_root)
    if index_root.exists():
        if not overwrite:
            print(f"  [cache] {index_root} exists, skipping build")
            return
        shutil.rmtree(index_root)
    index_root.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = min(len(docs_by_shard), os.cpu_count() or 1)
    n_workers = max(n_workers, 1)

    total = len(docs_by_shard)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_build_shard, shard_docs, shard_dir(index_root, shard_id))
            for shard_id, shard_docs in docs_by_shard.items()
        ]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()  # re-raise worker exceptions
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] shards built")


def index_size_mb(index_root: Path) -> float:
    """Total on-disk size of an index tree, in megabytes."""
    total = 0
    for dirpath, _, filenames in os.walk(index_root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if not os.path.islink(path):
                total += os.path.getsize(path)
    return total / 1024 / 1024
