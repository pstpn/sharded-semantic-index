"""Descriptive statistics of the collection, the graph and the clusterings."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd

from sharded_index.config import ClusteringConfig
from sharded_index.data.corpus import Corpus
from sharded_index.graph.clustering import modularity
from sharded_index.graph.cooccurrence import graph_terms, node_strength
from sharded_index.text import Tokenizer


def collection_description(
    pairs: pd.DataFrame,
    corpus: Corpus,
    train_queries: Sequence[str],
    holdout_queries: Sequence[str],
) -> dict[str, Any]:
    """Sizes of the collection and its vocabulary and the most frequent terms."""
    selected = pairs[pairs["is_selected"]]
    lengths = np.asarray(corpus.doc_terms.sum(axis=1)).ravel()
    share = corpus.df / corpus.n_docs
    head = sorted(
        zip(corpus.vocabulary, share.tolist(), strict=True), key=lambda item: (-item[1], item[0])
    )
    return {
        "collection": {
            "documents": corpus.n_docs,
            "judged_relevant_documents": int(selected["doc_id"].nunique()),
            "distractor_documents": int(corpus.n_docs - selected["doc_id"].nunique()),
            "queries": len(train_queries) + len(holdout_queries),
            "train_queries": len(train_queries),
            "holdout_queries": len(holdout_queries),
            "judged_per_query_mean": round(float(selected.groupby("query").size().mean()), 4),
        },
        "vocabulary": {
            "terms": len(corpus.vocabulary),
            "document_term_pairs": int(corpus.doc_terms.nnz),
            "terms_per_document_mean": round(float(lengths.mean()), 4),
            "terms_per_document_median": float(np.median(lengths)),
        },
        "document_frequency": {
            "max_share": round(head[0][1], 4),
            "top10": {term: round(value, 4) for term, value in head[:10]},
            "terms_over_1pct": int((share > 0.01).sum()),
            "terms_over_5pct": int((share > 0.05).sum()),
        },
    }


def graph_description(edges: pd.DataFrame, corpus: Corpus, frame_stop_words: int) -> dict[str, Any]:
    """Size and connectivity of the term graph and its coverage of the corpus vocabulary."""
    terms = graph_terms(edges)
    position = {term: i for i, term in enumerate(terms)}
    graph = ig.Graph(
        n=len(terms),
        edges=[(position[a], position[b]) for a, b in zip(edges["src"], edges["dst"], strict=True)],
    )
    components = graph.connected_components().sizes()
    in_corpus = len(set(terms) & set(corpus.vocabulary))
    return {
        "terms": len(terms),
        "edges": len(edges),
        "components": len(components),
        "largest_component_terms": max(components, default=0),
        "terms_in_corpus": in_corpus,
        "corpus_vocabulary_share": round(in_corpus / len(corpus.vocabulary), 4),
        "frame_stop_words": frame_stop_words,
    }


def clusterings_table(
    edges: pd.DataFrame,
    clusterings: Mapping[str, Mapping[str, int]],
    config: ClusteringConfig,
) -> pd.DataFrame:
    """One row per clustering method: its settings, cluster sizes and modularity."""
    rows = []
    for name, clustering in clusterings.items():
        method = config.methods[name]
        sizes = pd.Series(clustering).value_counts()
        rows.append(
            {
                "method": name,
                "algorithm": method.algorithm,
                "resolution": method.resolution,
                "n_parts": method.n_parts,
                "seed": config.seed,
                "graph_terms": len(clustering),
                "clusters": len(sizes),
                "largest_cluster": int(sizes.max()),
                "median_cluster": float(sizes.median()),
                "modularity": modularity(edges, dict(clustering)),
            }
        )
    return pd.DataFrame(rows)


def clusters_table(
    edges: pd.DataFrame,
    clustering: Mapping[str, int],
    top_clusters: int,
    top_terms: int,
) -> pd.DataFrame:
    """The largest clusters of a clustering with their strongest terms and internal density.

    ``density`` is the share of term pairs of the cluster connected by an edge.
    """
    strength = node_strength(edges)
    members: dict[int, list[str]] = {}
    for term, cluster in clustering.items():
        members.setdefault(cluster, []).append(term)
    src_cluster = edges["src"].map(clustering)
    internal = edges[src_cluster == edges["dst"].map(clustering)].assign(cluster=src_cluster)
    internal_edges = internal.groupby("cluster").size()

    largest = sorted(members, key=lambda cluster: (-len(members[cluster]), cluster))[:top_clusters]
    rows = []
    for cluster in largest:
        terms = members[cluster]
        ranked = sorted(terms, key=lambda term: (-strength.get(term, 0.0), term))
        n_edges = int(internal_edges.get(cluster, 0))
        possible = len(terms) * (len(terms) - 1) / 2
        rows.append(
            {
                "cluster": cluster,
                "terms": len(terms),
                "internal_edges": n_edges,
                "density": n_edges / possible if possible else np.nan,
                "top_terms": " ".join(ranked[:top_terms]),
            }
        )
    return pd.DataFrame(rows)


def frame_word_candidates(
    train_queries: Sequence[str],
    corpus: Corpus,
    tokenizer: Tokenizer,
    listed: Collection[str],
    top_n: int,
) -> pd.DataFrame:
    """Signals for curating the graph stop list, for the most frequent query terms.

    A question-frame word shapes the question rather than names its topic:
    it is frequent in queries (``query_share``), comparatively rare in
    documents (``document_share``, ``ratio`` of the two) and tends to open the
    question (``lead_share``: occurrences among the first two terms).
    ``listed`` marks the words of the configured graph stop list.  The table
    is a candidate list for manual curation: topical words score high too.
    """
    in_queries: Counter[str] = Counter()
    leading: Counter[str] = Counter()
    for query in train_queries:
        tokens = tokenizer(query)
        in_queries.update(set(tokens))
        leading.update(set(tokens[:2]))
    document_share = dict(zip(corpus.vocabulary, (corpus.df / corpus.n_docs).tolist(), strict=True))

    rows = []
    for term, count in sorted(in_queries.items(), key=lambda item: (-item[1], item[0]))[:top_n]:
        query_share = count / len(train_queries)
        rows.append(
            {
                "term": term,
                "query_share": query_share,
                "document_share": document_share.get(term, 0.0),
                "ratio": query_share / max(document_share.get(term, 0.0), 1e-9),
                "lead_share": leading[term] / count,
                "listed": term in listed,
            }
        )
    return pd.DataFrame(rows).sort_values("ratio", ascending=False, ignore_index=True)
