"""Semantic term sharding of a full-text inverted index.

The library builds term partitions from a query co-occurrence graph, routes
queries and documents to shards and measures what sharded search loses
against the unsharded index.  The DVC stages live in :mod:`sharded_index.pipeline`.
"""

__version__ = "0.2.0"
