"""Building Whoosh indices: the unsharded reference index and physical shard indices."""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import shutil
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path

from whoosh import index as whoosh_index
from whoosh.analysis import LowercaseFilter, RegexTokenizer, StopFilter
from whoosh.fields import ID, TEXT, Schema

from sharded_index.text import MIN_TOKEN_LEN

WRITER_MEMORY_MB = 256
FINGERPRINT_FILE = "CORPUS_FINGERPRINT"
"""Marker next to the reference index: the fingerprint of the corpus it was built from."""


def schema(stop_words: Collection[str]) -> Schema:
    """Index schema whose analyzer yields exactly the terms of the ``Tokenizer``."""
    analyzer = (
        RegexTokenizer()
        | LowercaseFilter()
        | StopFilter(stoplist=sorted(stop_words), minsize=MIN_TOKEN_LEN)
    )
    return Schema(doc_id=ID(stored=True, unique=True), text=TEXT(stored=True, analyzer=analyzer))


def build_index(documents: Mapping[str, str], target: Path, stop_words: Collection[str]) -> None:
    """Build one index of ``doc_id → text`` documents in ``target``, replacing what is there."""
    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    writer = whoosh_index.create_in(target, schema(stop_words)).writer(limitmb=WRITER_MEMORY_MB)
    for doc_id, text in documents.items():
        writer.add_document(doc_id=doc_id, text=text)
    writer.commit(merge=False)


def shard_path(root: Path, shard_id: int) -> Path:
    """Directory of one shard's index under ``root``."""
    return Path(root) / f"shard_{shard_id:04d}"


def build_shard_indices(
    shards: Mapping[int, Mapping[str, str]],
    root: Path,
    stop_words: Collection[str],
    n_workers: int | None = None,
) -> None:
    """Build an index per shard, the largest shards first.

    Shards are built by ``n_workers`` processes (all CPUs by default); with a
    single worker they are built in the calling process.
    """
    n_workers = max(n_workers or min(len(shards), os.cpu_count() or 1), 1)
    by_size = sorted(shards.items(), key=lambda item: -len(item[1]))
    if n_workers == 1:
        for shard_id, documents in by_size:
            build_index(documents, shard_path(root, shard_id), stop_words)
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(build_index, documents, shard_path(root, shard_id), frozenset(stop_words))
            for shard_id, documents in by_size
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def corpus_fingerprint(doc_ids: Iterable[str], stop_words: Collection[str]) -> str:
    """Content hash of what an index stores: the documents and the stop list.

    Document ids are hashes of the texts, so sorted ids identify the corpus.
    """
    digest = hashlib.md5()
    for doc_id in sorted(doc_ids):
        digest.update(doc_id.encode())
        digest.update(b";")
    digest.update(",".join(sorted(stop_words)).encode())
    return digest.hexdigest()
