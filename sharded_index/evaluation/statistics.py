"""Aggregation of per-query measurements: summaries and bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sharded_index.evaluation.measure import budget_labels


def routing_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Cover statistics of one strategy on one sample (evaluated queries only)."""
    rows = frame[frame["evaluated"]]
    fanout = rows["fanout"]
    hash_probes = rows["hash_probes"]
    has_hash_space = len(rows) > 0 and bool(hash_probes.notna().all())
    first_probes = np.bincount(rows["first_shard"].to_numpy(dtype=np.int64))
    return {
        "queries": len(frame),
        "evaluated": len(rows),
        "terms_mean": rows["terms"].mean(),
        "known_terms_mean": rows["known_terms"].mean(),
        "fanout_mean": fanout.mean(),
        "fanout_median": fanout.median(),
        "fanout_p95": fanout.quantile(0.95),
        "single_shard_share": (fanout == 1).mean(),
        "hash_probe_share": hash_probes.sum() / fanout.sum() if has_hash_space else np.nan,
        "mixed_query_share": (
            ((hash_probes > 0) & (hash_probes < fanout)).mean() if has_hash_space else np.nan
        ),
        "top_shard_traffic": first_probes.max() / first_probes.sum() if len(rows) else np.nan,
    }


def retrieval_summary(frame: pd.DataFrame, budgets: Sequence[int]) -> dict[str, dict[str, float]]:
    """Overlap and probed volume of one strategy on one sample, keyed by budget."""
    rows = frame[frame["evaluated"]]
    return {
        label: {
            "overlap": rows[f"overlap_{label}"].mean(),
            "volume": rows[f"volume_{label}"].mean(),
            "volume_median": rows[f"volume_{label}"].median(),
        }
        for label in budget_labels(budgets)
    }


def bootstrap_weights(n_queries: int, n_resamples: int, seed: int) -> np.ndarray:
    """Resampling counts ``[n_resamples, n_queries]``: how often each query is drawn."""
    draws = np.random.default_rng(seed).integers(0, n_queries, size=(n_resamples, n_queries))
    weights = np.empty((n_resamples, n_queries))
    for resample in range(n_resamples):
        weights[resample] = np.bincount(draws[resample], minlength=n_queries)
    return weights


def bootstrap_ci(
    values: np.ndarray, weights: np.ndarray, confidence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile confidence interval of the mean of every column of ``values``.

    ``values`` is ``[n_queries, n_metrics]``.  NaN marks a query outside the
    population: it is resampled with the others and left out of every mean,
    exactly as in the point estimate.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    valid = np.isfinite(values)
    means = (weights @ np.where(valid, values, 0.0)) / (weights @ valid.astype(np.float64))
    tail = round(50 * (1 - confidence), 10)
    return np.percentile(means, tail, axis=0), np.percentile(means, 100 - tail, axis=0)
