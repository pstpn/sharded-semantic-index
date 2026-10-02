"""Hash assignment of terms: the random baseline and the fallback for out-of-graph terms."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping


def hash_shard(term: str, n_shards: int) -> int:
    """Shard of a term by its MD5, stable across runs and platforms."""
    return int(hashlib.md5(term.encode()).hexdigest(), 16) % n_shards


def id_space(term_to_shard: Mapping[str, int]) -> int:
    """Size of the shard id space of an assignment (largest id + 1)."""
    return max(term_to_shard.values(), default=-1) + 1


def hash_partition(vocabulary: Iterable[str], n_shards: int) -> dict[str, int]:
    """Random baseline: every term goes to ``hash(term) % n_shards``."""
    return {term: hash_shard(term, n_shards) for term in sorted(vocabulary)}


def with_hash_fallback(clustering: Mapping[str, int], vocabulary: Iterable[str]) -> dict[str, int]:
    """Extend a clustering to the vocabulary by hashing the terms it does not cover.

    Out-of-graph terms go to a separate id space ``[n, 2n)``, where ``n`` is
    the id space of the clustering, so semantic shards hold graph terms only.
    """
    n = id_space(clustering)
    return {
        term: clustering[term] if term in clustering else n + hash_shard(term, max(n, 1))
        for term in sorted(vocabulary)
    }
