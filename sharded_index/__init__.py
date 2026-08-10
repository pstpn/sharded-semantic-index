"""Semantic term sharding for a full-text inverted index.

Core pipeline (exported here):

1. :mod:`.graph` — NPMI-weighted term co-occurrence graph from a corpus.
2. :mod:`.clustering` — Leiden partition of graph terms.
3. :mod:`.partition` — :class:`TermPartition` / :class:`ReplicatedTermPartition`:
   term → shard(s), routing and greedy query cover.
4. :mod:`.indexing` — one Whoosh BM25 index per shard.
5. :mod:`.search` — :class:`ShardedSearcher` fans a query out and merges results.

Import as submodules (heavier dependencies, notebook/pipeline layer):

- :mod:`.config` — artefact paths and ``params.yaml`` access;
- :mod:`.dataset` — MS MARCO extraction and views over the pairs;
- :mod:`.evaluation` — ground truth, routing recall, cover fanout;
- :mod:`.clusters` — tabular views of the Leiden clusters;
- :mod:`.plots` — all the figures;
- :mod:`.pipeline` — DVC stage entrypoints (``python -m sharded_index.pipeline.<stage>``).

Key invariant
-------------
Tokenization at routing time must match tokenization at index time:
:func:`tokenize` and the Whoosh analyzer both lowercase and drop the same
:data:`STOP_WORDS`.  The NPMI graph uses a slightly narrower vocabulary
(letters only, see :mod:`.graph`); terms outside the graph are covered by the
hash fallback of :meth:`TermPartition.with_hash_fallback`.
"""

from .clustering import cluster_terms
from .config import PartitionConfig
from .dataset import extract_query_doc_pairs
from .graph import build_npmi_graph, compute_node_strength
from .indexing import WHOOSH_ANALYZER, WHOOSH_SCHEMA, build_whoosh_indices, index_size_mb
from .partition import (
    ReplicatedTermPartition,
    TermPartition,
    assign_docs_to_shards,
    cluster_affinity,
    hash_term_to_shard,
    load_partition,
    replicate_with_volume_guard,
)
from .search import SearchResult, ShardedSearcher
from .text import STOP_WORDS, normalize_text, tokenize

__all__ = [
    "PartitionConfig",
    "ReplicatedTermPartition",
    "SearchResult",
    "ShardedSearcher",
    "STOP_WORDS",
    "TermPartition",
    "WHOOSH_ANALYZER",
    "WHOOSH_SCHEMA",
    "assign_docs_to_shards",
    "build_npmi_graph",
    "build_whoosh_indices",
    "cluster_affinity",
    "cluster_terms",
    "compute_node_strength",
    "extract_query_doc_pairs",
    "hash_term_to_shard",
    "index_size_mb",
    "load_partition",
    "replicate_with_volume_guard",
    "normalize_text",
    "tokenize",
]
