"""Slices of a query sample: by the connectivity of the query's terms and by its length."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from sharded_index.graph.cooccurrence import graph_terms
from sharded_index.text import Tokenizer

SLICE_COLUMNS = ["slicing", "slice", "position", "connectivity"]
OUT_OF_GRAPH = "out_of_graph"
SINGLE_TERM = "single_term"


def connectivity_slices(
    queries: Sequence[str],
    evaluated: Sequence[bool],
    edges: pd.DataFrame,
    tokenizer: Tokenizer,
    n_quantiles: int,
) -> pd.DataFrame:
    """Slice the evaluated queries by how strongly their own terms are connected in the graph.

    - ``out_of_graph`` — some term of the query is not a graph term;
    - ``single_term`` — a single graph term;
    - ``Q1..Qn`` — quantiles of the mean edge weight over all pairs of the
      query's terms (a pair without an edge counts as 0).
    """
    in_graph = set(graph_terms(edges))
    pair_weight = {
        frozenset((a, b)): float(weight)
        for a, b, weight in zip(edges["src"], edges["dst"], edges["weight"], strict=True)
    }

    rows = []
    for position, (query, kept) in enumerate(zip(queries, evaluated, strict=True)):
        if not kept:
            continue
        terms = sorted(set(tokenizer(query)))
        if any(term not in in_graph for term in terms):
            rows.append((OUT_OF_GRAPH, position, np.nan))
        elif len(terms) < 2:
            rows.append((SINGLE_TERM, position, np.nan))
        else:
            weights = [pair_weight.get(frozenset(pair), 0.0) for pair in combinations(terms, 2)]
            rows.append(("", position, float(np.mean(weights))))
    frame = pd.DataFrame(rows, columns=["slice", "position", "connectivity"])

    scored = frame["connectivity"].notna()
    if scored.any():
        levels = np.arange(1, n_quantiles) / n_quantiles
        bounds = np.quantile(frame.loc[scored, "connectivity"], levels)
        quantile = np.searchsorted(bounds, frame.loc[scored, "connectivity"], side="left") + 1
        frame.loc[scored, "slice"] = [f"Q{q}" for q in quantile]
    frame = frame.sort_values(["slice", "position"], ignore_index=True)
    return frame.assign(slicing="connectivity")[SLICE_COLUMNS]


def term_count_slices(
    n_terms: Sequence[int],
    evaluated: Sequence[bool],
    max_terms: int,
) -> pd.DataFrame:
    """Slice the evaluated queries by their number of distinct terms; the last slice is open."""
    rows = [
        (min(count, max_terms), position)
        for position, (count, kept) in enumerate(zip(n_terms, evaluated, strict=True))
        if kept
    ]
    frame = pd.DataFrame(rows, columns=["count", "position"]).sort_values(
        ["count", "position"], ignore_index=True
    )
    labels = frame["count"].astype(str).where(frame["count"] < max_terms, f"{max_terms}+")
    return frame.assign(slicing="terms", slice=labels, connectivity=np.nan)[SLICE_COLUMNS]
