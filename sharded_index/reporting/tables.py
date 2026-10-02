"""Result tables built from the per-query measurements of every method and strategy.

Every table is tidy: one row per configuration, the same metric names in
every file.  ``frames`` below is the concatenation of the per-query tables
of all methods with ``method``, ``strategy`` and ``sample`` columns.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sharded_index.config import ComparisonsConfig, Strategy
from sharded_index.evaluation.measure import budget_labels
from sharded_index.evaluation.statistics import bootstrap_ci, retrieval_summary, routing_summary

PRECISION = 4
COUNT_COLUMNS = frozenset(
    {
        "queries",
        "evaluated",
        "shards",
        "replicated_terms",
        "hash_terms",
        "graph_terms",
        "clusters",
        "largest_cluster",
        "terms",
        "internal_edges",
        "incomplete_covers",
        "queries_checked",
        "document_term_pairs",
        "shards_built",
        "documents_indexed",
        "scores_compared",
        "result_mismatches",
        "ranking_order_violations",
        "ranking_prefix_mismatches",
    }
)
EXACT_COLUMNS = frozenset({"max_score_difference"})

Groups = Mapping[tuple[str, str, str], pd.DataFrame]


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a table as CSV: counts as integers, other numbers rounded to ``PRECISION``."""
    table = frame.copy()
    for column in table.columns:
        if column in COUNT_COLUMNS:
            table[column] = table[column].astype("Int64")
        elif column not in EXACT_COLUMNS and pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].round(PRECISION)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def group_queries(frames: pd.DataFrame) -> dict[tuple[str, str, str], pd.DataFrame]:
    """Per-query rows of every ``(method, strategy, sample)`` in query order."""
    return {
        key: group.sort_values("position")
        for key, group in frames.groupby(["method", "strategy", "sample"], sort=False)
    }


def _metric_columns(budgets: Sequence[int]) -> list[str]:
    labels = budget_labels(budgets)
    return [
        "fanout",
        "single_shard_share",
        *(f"overlap_{label}" for label in labels),
        *(f"volume_{label}" for label in labels),
    ]


def _metric_values(group: pd.DataFrame, budgets: Sequence[int]) -> np.ndarray:
    """Per-query values of the metrics of :func:`_metric_columns`; NaN where not evaluated."""
    single = np.where(group["evaluated"], group["fanout"] == 1, np.nan)
    labels = budget_labels(budgets)
    columns = [f"overlap_{label}" for label in labels] + [f"volume_{label}" for label in labels]
    return np.column_stack([group["fanout"].to_numpy(), single, group[columns].to_numpy()])


def routing_table(
    groups: Groups,
    keys: Iterable[tuple[str, str, str]],
    weights: Mapping[str, np.ndarray],
    confidence: float,
) -> pd.DataFrame:
    """Cover statistics per ``(method, strategy, sample)`` with intervals of the two means."""
    rows = []
    for method, strategy, sample in keys:
        group = groups[method, strategy, sample]
        single = np.where(group["evaluated"], group["fanout"] == 1, np.nan)
        low, high = bootstrap_ci(
            np.column_stack([group["fanout"], single]), weights[sample], confidence
        )
        summary = routing_summary(group)
        rows.append(
            {
                "method": method,
                "strategy": strategy,
                "sample": sample,
                "queries": summary["queries"],
                "evaluated": summary["evaluated"],
                "terms_mean": summary["terms_mean"],
                "known_terms_mean": summary["known_terms_mean"],
                "fanout_mean": summary["fanout_mean"],
                "fanout_ci_low": low[0],
                "fanout_ci_high": high[0],
                "fanout_median": summary["fanout_median"],
                "fanout_p95": summary["fanout_p95"],
                "single_shard_share": summary["single_shard_share"],
                "single_shard_ci_low": low[1],
                "single_shard_ci_high": high[1],
                "hash_probe_share": summary["hash_probe_share"],
                "mixed_query_share": summary["mixed_query_share"],
                "top_shard_traffic": summary["top_shard_traffic"],
            }
        )
    return pd.DataFrame(rows)


def retrieval_table(
    groups: Groups,
    keys: Iterable[tuple[str, str, str]],
    budgets: Sequence[int],
    weights: Mapping[str, np.ndarray],
    confidence: float,
) -> pd.DataFrame:
    """Overlap and probed volume per ``(method, strategy, sample, budget)`` with intervals."""
    labels = budget_labels(budgets)
    columns = [f"overlap_{label}" for label in labels] + [f"volume_{label}" for label in labels]
    rows = []
    for method, strategy, sample in keys:
        group = groups[method, strategy, sample]
        low, high = bootstrap_ci(group[columns].to_numpy(), weights[sample], confidence)
        for i, (label, summary) in enumerate(retrieval_summary(group, budgets).items()):
            j = len(labels) + i
            rows.append(
                {
                    "method": method,
                    "strategy": strategy,
                    "sample": sample,
                    "budget": label,
                    "overlap": summary["overlap"],
                    "overlap_ci_low": low[i],
                    "overlap_ci_high": high[i],
                    "volume": summary["volume"],
                    "volume_ci_low": low[j],
                    "volume_ci_high": high[j],
                    "volume_median": summary["volume_median"],
                }
            )
    return pd.DataFrame(rows)


def comparison_pairs(
    methods: Sequence[str],
    strategies: Sequence[str],
    comparisons: ComparisonsConfig,
) -> list[tuple[str, tuple[str, str], tuple[str, str]]]:
    """The compared ``(method, strategy)`` pairs as ``(kind, subject, baseline)``.

    - ``hash`` — every strategy against the hash baseline of its family;
    - ``strategy`` — the configured pairs of strategies within a method;
    - ``method`` — every method against the reference method, strategy by strategy.
    """
    pairs = []
    for method in methods:
        for name in strategies:
            strategy = Strategy.parse(name)
            if not strategy.is_hash and strategy.hash_baseline in strategies:
                pairs.append(("hash", (method, name), (method, strategy.hash_baseline)))
        pairs += [("strategy", (method, a), (method, b)) for a, b in comparisons.pairs]
    reference = comparisons.reference_method
    if reference is not None:
        pairs += [
            ("method", (method, name), (reference, name))
            for method in methods
            if method != reference
            for name in strategies
        ]
    return pairs


def comparisons_table(
    groups: Groups,
    pairs: Iterable[tuple[str, tuple[str, str], tuple[str, str]]],
    samples: Sequence[str],
    budgets: Sequence[int],
    weights: Mapping[str, np.ndarray],
    confidence: float,
) -> pd.DataFrame:
    """Paired differences, subject minus baseline, over the same queries, with intervals."""
    metrics = _metric_columns(budgets)
    rows = []
    for kind, (method, strategy), (baseline_method, baseline_strategy) in pairs:
        for sample in samples:
            difference = _metric_values(groups[method, strategy, sample], budgets) - _metric_values(
                groups[baseline_method, baseline_strategy, sample], budgets
            )
            low, high = bootstrap_ci(difference, weights[sample], confidence)
            means = np.nanmean(difference, axis=0)
            rows += [
                {
                    "comparison": kind,
                    "method": method,
                    "strategy": strategy,
                    "baseline_method": baseline_method,
                    "baseline_strategy": baseline_strategy,
                    "sample": sample,
                    "metric": metric,
                    "difference": means[i],
                    "ci_low": low[i],
                    "ci_high": high[i],
                }
                for i, metric in enumerate(metrics)
            ]
    return pd.DataFrame(rows)


def variants_table(parts: Mapping[str, pd.DataFrame], leading: Sequence[str]) -> pd.DataFrame:
    """Per-method variant tables stacked into one, with ``leading`` columns first."""
    table = pd.concat(
        [part.assign(method=method) for method, part in parts.items()], ignore_index=True
    )
    if "value" in table:
        table["value"] = [
            str(int(value)) if float(value).is_integer() else str(value) for value in table["value"]
        ]
    return table[[*leading, *table.columns.drop(list(leading))]]


def slices_table(
    groups: Groups,
    keys: Iterable[tuple[str, str]],
    sample: str,
    slices: pd.DataFrame,
    budgets: Sequence[int],
) -> pd.DataFrame:
    """Metrics of every ``(method, strategy)`` on the slices of one sample.

    ``slices`` assigns query positions to slices (``[slicing, slice, position,
    connectivity]``); a query may belong to one slice of every slicing.
    """
    metrics = _metric_columns(budgets)
    rows = []
    for method, strategy in keys:
        group = groups[method, strategy, sample].set_index("position")
        values = pd.DataFrame(_metric_values(group, budgets), index=group.index, columns=metrics)
        for (slicing, name), members in slices.groupby(["slicing", "slice"], sort=False):
            positions = members["position"]
            rows.append(
                {
                    "method": method,
                    "strategy": strategy,
                    "slicing": slicing,
                    "slice": name,
                    "queries": len(positions),
                    "connectivity_min": members["connectivity"].min(),
                    "connectivity_max": members["connectivity"].max(),
                    "terms_mean": group.loc[positions, "terms"].mean(),
                    "known_terms_mean": group.loc[positions, "known_terms"].mean(),
                    **values.loc[positions].mean().rename({"fanout": "fanout_mean"}).to_dict(),
                }
            )
    return pd.DataFrame(rows)


def verification_table(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """One row per ``(method, strategy)`` from the per-method verification reports."""
    rows = []
    for method, result in results.items():
        shared = {
            "document_term_pairs": result["document_term_pairs"],
            "ranking_order_violations": result["reference"]["ranking_order_violations"],
            "ranking_prefix_mismatches": result["reference"]["ranking_prefix_mismatches"],
        }
        for strategy, checks in result["strategies"].items():
            completeness, physical = checks["completeness"], checks["physical"]
            rows.append(
                {
                    "method": method,
                    "strategy": strategy,
                    "assignment_exact": completeness["assignment_exact"],
                    "incomplete_covers": completeness["incomplete_covers"],
                    "queries_checked": completeness["queries_checked"],
                    **shared,
                    "shards_built": physical["shards_built"],
                    "documents_indexed": physical["documents_indexed"],
                    "scores_compared": physical["scores_compared"],
                    "max_score_difference": physical["max_score_difference"],
                    "result_mismatches": physical["result_mismatches"],
                    "passed": checks["passed"],
                }
            )
    return pd.DataFrame(rows)
