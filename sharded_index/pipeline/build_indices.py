"""Stage 3: assign documents to shards, build Whoosh indices per strategy + full index.

An index is rebuilt only when the document assignment actually changed:
``doc_id`` is the MD5 of the document text, so the sorted ``(doc_id, shards)``
pairs fully determine the index content.  Their hash is stored next to the
index (``ASSIGNMENT_FINGERPRINT``); on a match the build is skipped — DVC
keeps the directory between runs via ``persist: true``.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from sharded_index import (
    assign_docs_to_shards,
    build_whoosh_indices,
    index_size_mb,
    load_partition,
)
from sharded_index.config import (
    ASSIGNMENTS_DIR,
    INDICES_DIR,
    PAIRS_PATH,
    PARTITIONS_DIR,
    load_params,
    strategy_names,
)
from sharded_index.dataset import docs_from_pairs

FINGERPRINT_FILE = "ASSIGNMENT_FINGERPRINT"


def assignment_fingerprint(doc_shards: dict[str, set[int]]) -> str:
    """Content hash of an assignment — order-independent."""
    digest = hashlib.md5()
    for doc_id in sorted(doc_shards):
        digest.update(doc_id.encode())
        digest.update(",".join(map(str, sorted(doc_shards[doc_id]))).encode())
        digest.update(b";")
    return digest.hexdigest()


def build_if_changed(docs_by_shard: dict, doc_shards: dict, index_root) -> None:
    fingerprint = assignment_fingerprint(doc_shards)
    marker = index_root / FINGERPRINT_FILE
    if marker.exists() and marker.read_text().strip() == fingerprint:
        print(f"  [fingerprint] {index_root} matches assignment, skipping build")
        return
    build_whoosh_indices(docs_by_shard, index_root, overwrite=True)
    marker.write_text(fingerprint + "\n")


def main() -> None:
    strategies = strategy_names(load_params())
    docs = docs_from_pairs(pd.read_parquet(PAIRS_PATH))
    ASSIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)

    for strategy in strategies:
        partition = load_partition(PARTITIONS_DIR / strategy)
        doc_shards, docs_by_shard = assign_docs_to_shards(docs, partition)

        pd.DataFrame(
            [(doc_id, shard) for doc_id, shards in doc_shards.items() for shard in sorted(shards)],
            columns=["doc_id", "shard"],
        ).to_parquet(ASSIGNMENTS_DIR / f"{strategy}.parquet", index=False)

        duplication = sum(len(s) for s in doc_shards.values()) / len(doc_shards)
        print(f"{strategy}: {len(docs_by_shard)} shards used, duplication {duplication:.2f}x")
        build_if_changed(docs_by_shard, doc_shards, INDICES_DIR / strategy)

    full_assignment = {doc_id: {0} for doc_id in docs}
    build_if_changed({0: docs}, full_assignment, INDICES_DIR / "full")

    for name in (*strategies, "full"):
        print(f"{name}: {index_size_mb(INDICES_DIR / name):.1f} MB")


if __name__ == "__main__":
    main()
