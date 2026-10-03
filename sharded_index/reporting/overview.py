"""Overview figures: every clustering method and strategy on one canvas.

Colour and marker shape are the clustering method throughout; a hash
baseline is drawn hollow (and dashed where there are lines).  Figures with
one panel per query sample share their value axis, so samples are comparable
at a glance; figures that depend on the shard budget have one row per budget.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from sharded_index.config import Strategy
from sharded_index.evaluation.measure import FULL
from sharded_index.reporting.labels import Labels
from sharded_index.reporting.style import (
    DIVERGING,
    FAMILY_MARKERS,
    HASH_COLOR,
    INK_MUTED,
    PAGE_WIDTH,
    SURFACE,
    categorical_x,
    marker_style,
    method_colors,
    method_markers,
    save,
    strategy_line_style,
    themed,
)

REPLICA_AREA = {1: 18, 2: 34, 3: 54}
"""Scatter marker area by replica count: more replicas, larger marker."""
REPLICA_SIZE = {1: 4, 2: 5.5, 3: 7}
"""The same code for legend markers, in points."""
PANEL_COLUMNS = 4
ERRORBAR = {"elinewidth": 0.6, "capsize": 0, "linestyle": "", "markersize": 3.5}


def with_baselines(focus: Sequence[str]) -> list[str]:
    """The focus strategies, each followed by its hash baseline."""
    names: list[str] = []
    for name in focus:
        for candidate in (name, Strategy.parse(name).hash_baseline):
            if candidate not in names:
                names.append(candidate)
    return names


def _method_handles(
    colors: Mapping[str, str], markers: Mapping[str, str], lines: bool = False
) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            linestyle="-" if lines else "",
            marker=markers[name],
            color=color,
            markerfacecolor=color,
            label=name,
        )
        for name, color in colors.items()
    ]


def _fill_handles(text: Labels) -> list[Line2D]:
    """Legend entries for the fill convention: filled = semantic strategy, hollow = hash."""
    return [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            color=INK_MUTED,
            markerfacecolor=INK_MUTED,
            label=text("semantic_strategy"),
        ),
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            color=HASH_COLOR,
            markerfacecolor=SURFACE,
            label=text("hash_baseline"),
        ),
    ]


def _focus_handles(strategies: Sequence[str], lines: bool) -> list[Line2D]:
    """Legend entries naming the focus strategies: filled = semantic, hollow = its hash."""
    return [
        Line2D(
            [],
            [],
            linestyle=strategy_line_style(name) if lines else "",
            marker="o",
            color=INK_MUTED,
            markerfacecolor=SURFACE if Strategy.parse(name).is_hash else INK_MUTED,
            label=name,
        )
        for name in strategies
    ]


def _strategy_handles(strategies: Sequence[str]) -> list[Line2D]:
    """Legend entries of the strategies by family marker, replica size and hash fill."""
    handles = []
    for name in strategies:
        strategy = Strategy.parse(name)
        handles.append(
            Line2D(
                [],
                [],
                color=INK_MUTED,
                linestyle="",
                markersize=REPLICA_SIZE[min(strategy.replicas, 3)],
                label=name,
                **marker_style(name, INK_MUTED),
            )
        )
    return handles


def _panel_grid(n_panels: int, height: float, **subplot_kw: Any) -> tuple[Figure, list[Any], Any]:
    """A grid of panels four wide; the first unused cell (if any) is returned for the legend."""
    n_columns = min(PANEL_COLUMNS, n_panels)
    n_rows = ceil(n_panels / n_columns)
    figure = Figure(figsize=(PAGE_WIDTH, height * n_rows + 0.2), layout="constrained")
    axes = figure.subplots(n_rows, n_columns, squeeze=False, **subplot_kw).ravel().tolist()
    spare = None
    for axis in axes[n_panels:]:
        axis.axis("off")
        spare = spare or axis
    return figure, axes[:n_panels], spare


def _legend(figure: Figure, spare: Any, handles: Sequence[Line2D]) -> None:
    if spare is not None:
        spare.legend(handles=handles, loc="center left", borderaxespad=0)
    else:
        figure.legend(handles=handles, loc="outside lower center", ncol=max(len(handles), 1))


def _family_separators(axis: Any, strategies: Sequence[str]) -> None:
    families = [Strategy.parse(name).family for name in strategies]
    for i in range(1, len(families)):
        if families[i] != families[i - 1]:
            axis.axvline(i - 0.5, color="#dddddd", linewidth=0.5, zorder=0)


def _panel_title(text: Labels, index: int, sample: str, count: int) -> str:
    return f"{text.panel(index)} {text.sample(sample)}, n = {text.number(count)}"


@themed
def plot_metric_by_sample(
    table: pd.DataFrame,
    metric: str,
    interval: tuple[str, str] | None,
    strategies: Sequence[str],
    samples: Sequence[str],
    counts: Mapping[str, int],
    text: Labels,
    ylabel: str,
    path: Path,
    limits: tuple[float, float] | None = None,
) -> None:
    """One panel per query sample: a metric of every strategy under every method.

    ``table`` has one row per (method, strategy, sample) with the metric and,
    optionally, the bounds of its confidence interval named by ``interval``.
    """
    methods = list(dict.fromkeys(table["method"]))
    colors, markers = method_colors(methods), method_markers(methods)
    figure, axes, spare = _panel_grid(len(samples), height=2.2, sharey=True)
    x = np.arange(len(strategies))
    dodge = 0.7 / max(len(methods), 1)
    hollow = np.array([Strategy.parse(name).is_hash for name in strategies])
    for index, (axis, sample) in enumerate(zip(axes, samples, strict=True)):
        for k, method in enumerate(methods):
            rows = (
                table[(table["method"] == method) & (table["sample"] == sample)]
                .set_index("strategy")
                .reindex(strategies)
            )
            values = rows[metric].to_numpy(dtype=float)
            offset = (k - (len(methods) - 1) / 2) * dodge
            for mask, face in ((~hollow, colors[method]), (hollow, SURFACE)):
                if not mask.any():
                    continue
                error = None
                if interval is not None:
                    low = rows[interval[0]].to_numpy(dtype=float)[mask]
                    high = rows[interval[1]].to_numpy(dtype=float)[mask]
                    error = [values[mask] - low, high - values[mask]]
                axis.errorbar(
                    x[mask] + offset,
                    values[mask],
                    yerr=error,
                    marker=markers[method],
                    color=colors[method],
                    markerfacecolor=face,
                    **ERRORBAR,
                )
        _family_separators(axis, strategies)
        axis.set_xticks(x, strategies, rotation=60, ha="right", rotation_mode="anchor")
        categorical_x(axis)
        axis.set_xlim(-0.6, len(strategies) - 0.4)
        axis.set_title(_panel_title(text, index, sample, counts[sample]), loc="left")
        if limits is not None:
            axis.set_ylim(*limits)
    for axis in axes[::PANEL_COLUMNS]:
        axis.set_ylabel(ylabel)
    _legend(figure, spare, _method_handles(colors, markers) + _fill_handles(text))
    save(figure, path)


@themed
def plot_gain_by_sample(
    comparisons: pd.DataFrame,
    metric: str,
    strategies: Sequence[str],
    samples: Sequence[str],
    counts: Mapping[str, int],
    text: Labels,
    xlabel: str,
    path: Path,
) -> None:
    """One panel per sample: paired difference of every semantic strategy against its hash baseline.

    Intervals are the bootstrap confidence intervals of the comparisons table.
    """
    rows = comparisons[
        (comparisons["comparison"] == "hash")
        & (comparisons["metric"] == metric)
        & comparisons["strategy"].isin(strategies)
        & comparisons["sample"].isin(samples)
    ]
    methods = list(dict.fromkeys(rows["method"]))
    colors, markers = method_colors(methods), method_markers(methods)
    names = [name for name in strategies if name in set(rows["strategy"])]
    figure, axes, spare = _panel_grid(len(samples), height=2.0, sharex=True, sharey=True)
    y = np.arange(len(names))[::-1]
    dodge = 0.7 / max(len(methods), 1)
    for index, (axis, sample) in enumerate(zip(axes, samples, strict=True)):
        panel = rows[rows["sample"] == sample]
        for k, method in enumerate(methods):
            series = panel[panel["method"] == method].set_index("strategy").reindex(names)
            difference = series["difference"].to_numpy(dtype=float)
            axis.errorbar(
                difference,
                y + (k - (len(methods) - 1) / 2) * dodge,
                xerr=[
                    difference - series["ci_low"].to_numpy(dtype=float),
                    series["ci_high"].to_numpy(dtype=float) - difference,
                ],
                marker=markers[method],
                color=colors[method],
                markerfacecolor=colors[method],
                **ERRORBAR,
            )
        axis.axvline(0, color=INK_MUTED, linewidth=0.6, zorder=1)
        axis.set_yticks(y, names)
        axis.tick_params(axis="y", which="minor", left=False, right=False)
        axis.set_title(_panel_title(text, index, sample, counts[sample]), loc="left")
    figure.supxlabel(xlabel, fontsize=8)
    _legend(figure, spare, _method_handles(colors, markers))
    save(figure, path)


@themed
def plot_quality_vs_cost(
    points: Mapping[int, pd.DataFrame], top_k: int, text: Labels, path: Path
) -> None:
    """Overlap against the three costs — storage, routing, probed volume — one row per budget.

    ``points[budget]`` has one row per (method, strategy) with ``duplication``,
    ``fanout_mean``, ``volume`` and ``overlap`` at that budget.
    """
    budgets = list(points)
    first = points[budgets[0]]
    methods = list(dict.fromkeys(first["method"]))
    colors = method_colors(methods)
    costs = [
        ("duplication", text("duplication_short")),
        ("fanout_mean", text("fanout_short")),
        ("volume", text("volume")),
    ]
    figure = Figure(figsize=(PAGE_WIDTH, 2.2 * len(budgets) + 0.5), layout="constrained")
    grid = figure.subplots(len(budgets), len(costs), sharex="col", sharey="row", squeeze=False)
    for i, budget in enumerate(budgets):
        for j, (column, label) in enumerate(costs):
            axis = grid[i, j]
            for row in points[budget].itertuples(index=False):
                strategy = Strategy.parse(row.strategy)
                color = colors[row.method]
                axis.scatter(
                    getattr(row, column),
                    row.overlap,
                    s=REPLICA_AREA[min(strategy.replicas, 3)],
                    marker=FAMILY_MARKERS[strategy.family],
                    facecolors=SURFACE if strategy.is_hash else color,
                    edgecolors=color,
                    linewidths=0.8,
                    zorder=3,
                )
            axis.set_xlim(left=0)
            if i == len(budgets) - 1:
                axis.set_xlabel(label)
        grid[i, 0].set_ylabel(text("overlap_at", k=top_k, shards=text.shards(budget)))
    strategies = list(dict.fromkeys(first["strategy"]))
    handles = [
        Line2D([], [], linestyle="", marker="o", color=color, markerfacecolor=color, label=name)
        for name, color in colors.items()
    ] + _strategy_handles(strategies)
    figure.legend(handles=handles, loc="outside lower center", ncol=7)
    save(figure, path)


@themed
def plot_budget_curves(
    retrieval: pd.DataFrame, focus: Sequence[str], top_k: int, text: Labels, path: Path
) -> None:
    """Overlap and probed volume at every shard budget: focus strategies against their baselines.

    ``retrieval`` holds the rows of one sample of the retrieval table.
    """
    strategies = with_baselines(focus)
    methods = list(dict.fromkeys(retrieval["method"]))
    colors, markers = method_colors(methods), method_markers(methods)
    budgets = list(dict.fromkeys(retrieval["budget"]))
    positions = list(range(len(budgets)))
    figure = Figure(figsize=(PAGE_WIDTH, 2.5), layout="constrained")
    overlap_axis, volume_axis = figure.subplots(1, 2)
    for method in methods:
        for name in strategies:
            series = (
                retrieval[(retrieval["method"] == method) & (retrieval["strategy"] == name)]
                .set_index("budget")
                .reindex(budgets)
            )
            style = {
                "color": colors[method],
                "linestyle": strategy_line_style(name),
                **marker_style(name, colors[method], markers[method]),
            }
            overlap_axis.plot(positions, series["overlap"], **style)
            volume_axis.plot(positions, series["volume"], **style)
    labels = [text("full_cover") if budget == FULL else budget for budget in budgets]
    for axis in (overlap_axis, volume_axis):
        axis.set_xticks(positions, labels)
        categorical_x(axis)
        axis.set_xlabel(text("shards_probed"))
    overlap_axis.set_ylabel(text("overlap_at_budget", k=top_k))
    volume_axis.set_ylabel(text("volume"))
    handles = _method_handles(colors, markers, lines=True) + _focus_handles(strategies, lines=True)
    figure.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    save(figure, path)


@themed
def plot_slices(
    slices: pd.DataFrame,
    focus: Sequence[str],
    top_k: int,
    budgets: Sequence[int],
    text: Labels,
    path: Path,
) -> None:
    """Overlap of the focus strategies and baselines on every query slice.

    Columns are the slicings (connectivity, length), rows the shard budgets.
    """
    strategies = with_baselines(focus)
    methods = list(dict.fromkeys(slices["method"]))
    colors, markers = method_colors(methods), method_markers(methods)
    slicings = list(dict.fromkeys(slices["slicing"]))
    figure = Figure(figsize=(PAGE_WIDTH, 2.3 * len(budgets) + 0.5), layout="constrained")
    grid = figure.subplots(len(budgets), len(slicings), sharex="col", sharey="row", squeeze=False)
    dodge = 0.7 / max(len(methods) * len(strategies), 1)
    for i, budget in enumerate(budgets):
        for j, slicing in enumerate(slicings):
            axis = grid[i, j]
            part = slices[slices["slicing"] == slicing]
            names = list(dict.fromkeys(part["slice"]))
            counts = part.drop_duplicates("slice").set_index("slice")["queries"]
            x = np.arange(len(names))
            series_index = 0
            for method in methods:
                for name in strategies:
                    series = (
                        part[(part["method"] == method) & (part["strategy"] == name)]
                        .set_index("slice")
                        .reindex(names)
                    )
                    offset = (series_index - (len(methods) * len(strategies) - 1) / 2) * dodge
                    series_index += 1
                    axis.plot(
                        x + offset,
                        series[f"overlap_{budget}"],
                        linestyle="",
                        color=colors[method],
                        **marker_style(name, colors[method], markers[method]),
                    )
            for boundary in range(1, len(names)):
                axis.axvline(boundary - 0.5, color="#dddddd", linewidth=0.5, zorder=0)
            axis.set_xticks(
                x,
                [
                    f"{text.slice(slicing, name)}\n{text.number(int(counts[name]))}"
                    for name in names
                ],
            )
            categorical_x(axis)
            axis.set_xlim(-0.5, len(names) - 0.5)
            if i == len(budgets) - 1:
                axis.set_xlabel(text(slicing) if slicing in ("connectivity", "terms") else slicing)
        grid[i, 0].set_ylabel(text("overlap_at", k=top_k, shards=text.shards(budget)))
    handles = _method_handles(colors, markers) + _focus_handles(strategies, lines=False)
    figure.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    save(figure, path)


@themed
def plot_ablation_heatmaps(
    ablations: pd.DataFrame,
    strategies: Sequence[str],
    top_k: int,
    budget: int,
    text: Labels,
    path: Path,
) -> None:
    """Change of overlap against the baseline variant in percentage points, a panel per strategy."""
    methods = list(dict.fromkeys(ablations["method"]))
    variants = [name for name in dict.fromkeys(ablations["variant"]) if name != "baseline"]
    figure = Figure(figsize=(PAGE_WIDTH, 0.21 * len(variants) + 1.0), layout="constrained")
    axes = figure.subplots(1, len(strategies), sharey=True, squeeze=False)[0]
    for index, (axis, strategy) in enumerate(zip(axes, strategies, strict=True)):
        table = (
            ablations[ablations["strategy"] == strategy]
            .pivot(index="variant", columns="method", values=f"overlap_{budget}")
            .reindex(index=["baseline", *variants], columns=methods)
        )
        delta = (table.loc[variants] - table.loc["baseline"]) * 100
        values = delta.to_numpy(dtype=float)
        finite = np.isfinite(values)
        limit = max(float(np.nanmax(np.abs(values))) if finite.any() else 0.0, 0.5)
        image = axis.imshow(values, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if finite[i, j]:
                    strong = abs(values[i, j]) > 0.6 * limit
                    axis.text(
                        j,
                        i,
                        f"{values[i, j]:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=SURFACE if strong else "black",
                    )
        axis.set_xticks(range(len(methods)), methods)
        axis.set_yticks(range(len(variants)), variants)
        axis.set_xticks(np.arange(-0.5, len(methods)), minor=True)
        axis.set_yticks(np.arange(-0.5, len(variants)), minor=True)
        axis.grid(visible=True, which="minor", color=SURFACE, linewidth=1.5)
        axis.tick_params(which="both", length=0)
        axis.set_title(f"{text.panel(index)} {strategy}", loc="left")
        figure.colorbar(
            image,
            ax=axis,
            shrink=0.9,
            label=text("delta_pp", k=top_k, shards=text.shards(budget)),
        )
    save(figure, path)
