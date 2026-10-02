"""Evaluation query samples: real query logs and synthetic queries from the graph."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from sharded_index.config import ARITY_NAMES, QueriesConfig
from sharded_index.text import normalize_text

QUERY_COLUMNS = ["sample", "position", "query"]


def sample_queries(queries: list[str], size: int | None, seed: int) -> list[str]:
    """A seeded random sample in the original order; all queries if ``size`` covers them."""
    if size is None or len(queries) <= size:
        return list(queries)
    chosen = np.sort(np.random.default_rng(seed).choice(len(queries), size=size, replace=False))
    return [queries[i] for i in chosen]


def load_external_queries(dataset: str, config: str | None, split: str, column: str) -> list[str]:
    """Normalized, deduplicated queries of a HuggingFace dataset."""
    from datasets import load_dataset  # noqa: PLC0415 - heavy optional import

    rows = load_dataset(dataset, config, split=split)
    normalized = (normalize_text(text) for text in rows[column])
    return [query for query in dict.fromkeys(normalized) if query]


def _ranked_edges(edges: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Edges ordered by association strength (``connected``) or by co-occurrence count."""
    keys = ["weight", "count"] if kind == "connected" else ["count", "weight"]
    return edges.sort_values([*keys, "src", "dst"], ascending=[False, False, True, True])


def _triples(ranked: pd.DataFrame, links: dict[str, dict[str, float]], size: int) -> list[str]:
    """Each pair extended by the common neighbour with the strongest weaker link."""
    triples: list[str] = []
    seen: set[frozenset[str]] = set()
    for a, b in zip(ranked["src"], ranked["dst"], strict=True):
        common = set(links[a]) & set(links[b])
        if not common:
            continue
        third = max(common, key=lambda term: (min(links[a][term], links[b][term]), term))
        terms = frozenset((a, b, third))
        if terms not in seen:
            seen.add(terms)
            triples.append(f"{a} {b} {third}")
        if len(triples) == size:
            break
    return triples


def synthetic_queries(edges: pd.DataFrame, kind: str, arity: int, size: int) -> list[str]:
    """Pseudo-queries made of graph terms, independent of any clustering.

    ``connected`` takes the edges with the highest weight (terms that occur
    together almost exclusively), ``frequent`` the edges with the most
    co-occurrences.  A triple is a pair plus their best common neighbour.
    """
    ranked = _ranked_edges(edges, kind)
    if arity == 2:
        top = ranked.head(size)
        return [f"{a} {b}" for a, b in zip(top["src"], top["dst"], strict=True)]

    measure = "weight" if kind == "connected" else "count"
    links: dict[str, dict[str, float]] = defaultdict(dict)
    for a, b, value in zip(edges["src"], edges["dst"], edges[measure], strict=True):
        links[a][b] = links[b][a] = float(value)
    return _triples(ranked, links, size)


def build_samples(
    train_queries: list[str],
    holdout_queries: list[str],
    edges: pd.DataFrame,
    config: QueriesConfig,
) -> pd.DataFrame:
    """All evaluation samples as one ``[sample, position, query]`` table."""
    samples = {
        "train": sample_queries(train_queries, config.train_size, config.sample_seed),
        "holdout": sample_queries(holdout_queries, config.holdout_size, config.sample_seed),
    }
    if config.ood is not None:
        ood = config.ood
        external = load_external_queries(ood.dataset, ood.config, ood.split, ood.column)
        samples["ood"] = sample_queries(external, ood.size, config.sample_seed)
    if config.synthetic is not None:
        for kind in config.synthetic.kinds:
            for arity in config.synthetic.arities:
                name = f"{kind}_{ARITY_NAMES[arity]}"
                samples[name] = synthetic_queries(edges, kind, arity, config.synthetic.size)

    frames = [
        pd.DataFrame({"sample": name, "position": range(len(queries)), "query": queries})
        for name, queries in samples.items()
    ]
    return pd.concat(frames, ignore_index=True)[QUERY_COLUMNS]


def queries_of(samples: pd.DataFrame, sample: str) -> list[str]:
    """Queries of one sample in their stored order."""
    rows = samples[samples["sample"] == sample].sort_values("position")
    return rows["query"].tolist()
