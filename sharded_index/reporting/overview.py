"""Overview figures: every clustering method on one canvas.

Colour and marker shape are the clustering method throughout; a hash
baseline is drawn hollow (and dashed where there are lines).  The figures of
the detailed comparison show only the focus strategies and their hash
baselines; the figures of the unbudgeted regime show every strategy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from textwrap import fill
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from sharded_index.config import Strategy
from sharded_index.evaluation.measure import FULL
from sharded_index.reporting.labels import Labels
from sharded_index.reporting.style import (
    COLUMN_WIDTH,
    DIVERGING,
    FAMILY_MARKERS,
    GRID_COLOR,
    INK_MUTED,
    PAGE_WIDTH,
    SEQUENTIAL,
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
ERRORBAR = {"elinewidth": 0.6, "capsize": 0, "linestyle": "", "markersize": 3.5}
SEPARATOR = {"color": "#dddddd", "linewidth": 0.5, "zorder": 0}


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


def _strategy_handles(strategies: Sequence[str], lines: bool = False) -> list[Line2D]:
    """Legend entries of the strategies by family marker, replica size and hash fill."""
    handles = []
    for name in strategies:
        strategy = Strategy.parse(name)
        handles.append(
            Line2D(
                [],
                [],
                color=INK_MUTED,
                linestyle=strategy_line_style(name) if lines else "",
                markersize=REPLICA_SIZE[min(strategy.replicas, 3)],
                label=name,
                **marker_style(name, INK_MUTED),
            )
        )
    return handles


METHOD_NAMES = {"leiden": "Leiden", "cpm": "CPM", "infomap": "Infomap", "metis": "METIS"}
"""Display names of the clustering methods in the print figures."""
INK = "#000000"
PRINT_WIDTH = 6.7
"""Text width of the journal page (171 mm) in inches: figures drawn at final size."""


def _dumbbells(
    axis: Any,
    x: np.ndarray,
    pairs: Sequence[tuple[str, np.ndarray, np.ndarray]],
    markers: Mapping[str, str],
    colors: Mapping[str, str],
    width: float = 0.72,
) -> None:
    """Paired values per category, one column per method: hollow = baseline, filled = method.

    ``pairs`` holds ``(method, semantic, baseline)`` arrays aligned with ``x``; a thin
    segment in the method's colour joins the two values, so the size and sign of the
    gain read at a glance.  Methods differ by colour and marker shape.
    """
    step = width / max(len(pairs), 1)
    for k, (method, semantic, baseline) in enumerate(pairs):
        position = x + (k - (len(pairs) - 1) / 2) * step
        color = colors[method]
        axis.vlines(position, baseline, semantic, color=color, linewidth=0.9, alpha=0.55, zorder=2)
        axis.plot(
            position,
            baseline,
            linestyle="",
            marker=markers[method],
            markersize=4.2,
            markerfacecolor=SURFACE,
            markeredgecolor=color,
            markeredgewidth=0.9,
            zorder=3,
        )
        axis.plot(
            position,
            semantic,
            linestyle="",
            marker=markers[method],
            markersize=4.2,
            markerfacecolor=color,
            markeredgecolor=color,
            zorder=4,
        )
    for boundary in range(1, len(x)):
        axis.axvline(boundary - 0.5, color=GRID_COLOR, linewidth=0.5, zorder=0)
    axis.grid(visible=False, axis="x")
    axis.set_xlim(-0.5, len(x) - 0.5)
    categorical_x(axis)


def _dumbbell_legend(
    figure: Figure,
    methods: Sequence[str],
    markers: Mapping[str, str],
    colors: Mapping[str, str],
    semantic: str,
    baseline: str,
) -> None:
    """Methods by colour and shape, strategies by fill: all shapes filled or all hollow."""

    def marker(shape: str, face: str, edge: str = INK) -> Line2D:
        return Line2D(
            [],
            [],
            linestyle="",
            marker=shape,
            markersize=4.2,
            markerfacecolor=face,
            markeredgecolor=edge,
            markeredgewidth=0.8,
        )

    shapes = [markers[m] for m in methods]
    handles: list[Any] = [marker(markers[m], colors[m], colors[m]) for m in methods]
    handles += [
        tuple(marker(markers[m], colors[m], colors[m]) for m in methods),
        tuple(marker(markers[m], SURFACE, colors[m]) for m in methods),
    ]
    labels = [METHOD_NAMES.get(m, m) for m in methods] + [semantic, baseline]
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(handles),
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
        handlelength=4.0 if len(shapes) > 2 else 2.0,
    )


@themed
def plot_by_sample(
    routing: pd.DataFrame,
    focus: Sequence[str],
    samples: Sequence[str],
    counts: Mapping[str, int],
    text: Labels,
    path: Path,
) -> None:
    """Single-shard share (a) and mean cover size (b) on every query sample.

    The first focus strategy against its hash baseline under every method, drawn as
    paired values (see :func:`_dumbbells`).
    """
    semantic = focus[0]
    baseline = Strategy.parse(semantic).hash_baseline
    methods = list(dict.fromkeys(routing["method"]))
    markers, colors = method_markers(methods), method_colors(methods)
    panels = [
        ("single_shard_share", text("single_shard_share"), (0.0, 1.03)),
        ("fanout_mean", text("fanout_mean"), None),
    ]
    figure = Figure(figsize=(PRINT_WIDTH, 4.3), layout="constrained")
    axes = figure.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
    x = np.arange(len(samples))

    def values(method: str, name: str, metric: str) -> np.ndarray:
        rows = routing[(routing["method"] == method) & (routing["strategy"] == name)]
        return rows.set_index("sample").reindex(samples)[metric].to_numpy(dtype=float)

    for index, (axis, (metric, label, limits)) in enumerate(zip(axes, panels, strict=True)):
        pairs = [(m, values(m, semantic, metric), values(m, baseline, metric)) for m in methods]
        _dumbbells(axis, x, pairs, markers, colors)
        axis.set_ylabel(fill(label, 28))
        axis.set_title(text.panel(index), loc="left")
        if limits is not None:
            axis.set_ylim(*limits)
    axes[-1].set_xticks(
        x, [f"{text.sample(name)}\nn = {text.number(counts[name])}" for name in samples]
    )
    _dumbbell_legend(figure, methods, markers, colors, semantic, baseline)
    save(figure, path)


@themed
def plot_fanout_ecdf(
    fanouts: Mapping[str, Mapping[str, np.ndarray]], text: Labels, path: Path
) -> None:
    """Empirical CDF of the cover size: ``fanouts[method][strategy]`` are per-query values.

    The value at 1 is the single-shard share; the strategies are the focus
    strategies and their hash baselines (dashed).
    """
    methods = list(fanouts)
    strategies = list(dict.fromkeys(name for method in methods for name in fanouts[method]))
    colors, markers = method_colors(methods), method_markers(methods)
    figure = Figure(figsize=(PAGE_WIDTH, 2.8), layout="constrained")
    axis = figure.subplots()
    upper = 1
    for method in methods:
        for name, values in fanouts[method].items():
            ordered = np.sort(np.asarray(values, dtype=float))
            share = np.arange(1, len(ordered) + 1) / len(ordered)
            axis.step(
                ordered,
                share,
                where="post",
                color=colors[method],
                linestyle=strategy_line_style(name),
                linewidth=1.0,
            )
            upper = max(upper, int(np.percentile(ordered, 99)))
    axis.set_xlim(0.5, upper + 0.5)
    axis.set_ylim(0, 1.02)
    axis.set_xticks(range(1, upper + 1))
    categorical_x(axis)
    axis.set_xlabel(text("fanout"))
    axis.set_ylabel(text("share_of_queries"))
    handles = _method_handles(colors, markers, lines=True) + _focus_handles(strategies, lines=True)
    axis.legend(handles=handles, loc="lower right", ncol=2)
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
        grid[i, 0].set_ylabel(fill(text("overlap_at", k=top_k, shards=text.shards(budget)), 26))
    strategies = list(dict.fromkeys(first["strategy"]))
    handles = [
        Line2D([], [], linestyle="", marker="o", color=color, markerfacecolor=color, label=name)
        for name, color in colors.items()
    ] + _strategy_handles(strategies)
    figure.legend(handles=handles, loc="outside lower center", ncol=7)
    save(figure, path)


@themed
def plot_parameter_sweeps(
    sensitivity: pd.DataFrame,
    strategies: Sequence[str],
    volume_cap: float,
    budget: int,
    top_k: int,
    text: Labels,
    path: Path,
) -> None:
    """Largest shard, duplication and overlap against the algorithm's own parameter.

    One column per method, one line per strategy; the dotted line is the
    shard-volume budget.  A method without a parameter sweep in
    ``sensitivity`` (``sweep`` other than the seed and the training-log size)
    gets a single point: its main configuration, the full-log row of the
    training-size sweep.
    """
    rows_of = sensitivity[sensitivity["strategy"].isin(strategies)]
    methods = list(dict.fromkeys(rows_of["method"]))
    colors = method_colors(methods)
    rows: list[tuple[str, str, bool]] = [
        ("largest_shard", text("largest_shard_short"), True),
        ("duplication", text("duplication_short"), False),
        (f"overlap_{budget}", text("overlap_at", k=top_k, shards=text.shards(budget)), False),
    ]
    figure = Figure(figsize=(PAGE_WIDTH, 1.75 * len(rows) + 0.7), layout="constrained")
    grid = figure.subplots(len(rows), len(methods), sharex="col", sharey="row", squeeze=False)
    for j, method in enumerate(methods):
        of_method = rows_of[rows_of["method"] == method]
        part = of_method[~of_method["sweep"].isin(("seed", "train_size"))]
        if part.empty:
            full_log = of_method[of_method["sweep"] == "train_size"]
            part = full_log[full_log["value"] == full_log["value"].max()].assign(value=0.0)
            parameter, logarithmic = "", False
        else:
            parameter = str(part["sweep"].iloc[0])
            values = part["value"].to_numpy(dtype=float)
            logarithmic = values.max() / max(values.min(), 1e-9) >= 10
        for i, (column, label, with_budget) in enumerate(rows):
            axis = grid[i, j]
            for name in strategies:
                series = part[part["strategy"] == name].sort_values("value")
                axis.plot(
                    series["value"],
                    series[column],
                    color=colors[method],
                    linestyle=strategy_line_style(name),
                    linewidth=0.9,
                    markersize=3.5,
                    **marker_style(name, colors[method]),
                )
            if with_budget:
                axis.axhline(volume_cap, color=INK_MUTED, linestyle=":", linewidth=0.7)
            if logarithmic:
                axis.set_xscale("log")
            if not parameter:
                axis.set_xticks([0.0], ["—"])
                axis.set_xlim(-1, 1)
                categorical_x(axis)
            if i == 0:
                axis.set_title(f"{text.panel(j)} {method}", loc="left")
            if i == len(rows) - 1:
                axis.set_xlabel(
                    text(parameter)
                    if parameter in ("resolution", "n_parts")
                    else text("no_parameter")
                )
            if j == 0:
                axis.set_ylabel(fill(label, 24))
    handles = [
        *_strategy_handles(strategies, lines=True),
        Line2D([], [], color=INK_MUTED, linestyle=":", label=text("budget_line")),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=len(handles))
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
    for index, axis in enumerate((overlap_axis, volume_axis)):
        axis.set_xticks(positions, labels)
        categorical_x(axis)
        axis.set_xlabel(text("shards_probed"))
        axis.set_title(text.panel(index), loc="left")
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
    """Overlap of the first focus strategy against its hash baseline on every query slice.

    Columns are the slicings (connectivity, length), rows the shard budgets; every
    method is a pair of values (see :func:`_dumbbells`).
    """
    semantic = focus[0]
    baseline = Strategy.parse(semantic).hash_baseline
    methods = list(dict.fromkeys(slices["method"]))
    markers, colors = method_markers(methods), method_colors(methods)
    slicings = list(dict.fromkeys(slices["slicing"]))
    figure = Figure(figsize=(PRINT_WIDTH, 2.05 * len(budgets) + 0.75), layout="constrained")
    grid = figure.subplots(
        len(budgets),
        len(slicings),
        sharex="col",
        sharey="row",
        squeeze=False,
        gridspec_kw={
            "width_ratios": [
                len(set(slices.loc[slices["slicing"] == s, "slice"])) for s in slicings
            ]
        },
    )
    for i, budget in enumerate(budgets):
        for j, slicing in enumerate(slicings):
            axis = grid[i, j]
            part = slices[slices["slicing"] == slicing]
            names = list(dict.fromkeys(part["slice"]))
            counts = part.drop_duplicates("slice").set_index("slice")["queries"]

            column = f"overlap_{budget}"
            table = part.pivot_table(index="slice", columns=["method", "strategy"], values=column)
            table = table.reindex(names)
            pairs = [
                (
                    m,
                    table[(m, semantic)].to_numpy(dtype=float),
                    table[(m, baseline)].to_numpy(dtype=float),
                )
                for m in methods
            ]
            _dumbbells(axis, np.arange(len(names)), pairs, markers, colors)
            axis.set_xticks(
                np.arange(len(names)),
                [f"{text.slice(slicing, n)}\n{text.number(int(counts[n]))}" for n in names],
            )
            if i == 0:
                axis.set_title(text.panel(j), loc="left")
            if i == len(budgets) - 1:
                axis.set_xlabel(text(slicing) if slicing in ("connectivity", "terms") else slicing)
        grid[i, 0].set_ylabel(fill(text("overlap_at", k=top_k, shards=text.shards(budget)), 24))
    _dumbbell_legend(figure, methods, markers, colors, semantic, baseline)
    save(figure, path)


@themed
def plot_variant_heatmap(
    ablations: pd.DataFrame,
    strategy: str,
    metric: str,
    kind: Literal["delta", "value"],
    label: str,
    path: Path,
) -> None:
    """Heat map of one metric of one strategy over the configuration variants and the methods.

    ``delta`` shows the change against the ``baseline`` variant in percentage
    points on a diverging scale; ``value`` shows the metric itself in per cent
    on a sequential scale, baseline row included.
    """
    methods = list(dict.fromkeys(ablations["method"]))
    variants = [name for name in dict.fromkeys(ablations["variant"]) if name != "baseline"]
    table = (
        ablations[ablations["strategy"] == strategy]
        .pivot(index="variant", columns="method", values=metric)
        .reindex(index=["baseline", *variants], columns=methods)
    )
    if kind == "delta":
        names = variants
        values = ((table.loc[variants] - table.loc["baseline"]) * 100).to_numpy(dtype=float)
        finite = np.isfinite(values)
        limit = max(float(np.nanmax(np.abs(values))) if finite.any() else 0.0, 0.5)
        scale: dict[str, Any] = {"cmap": DIVERGING, "vmin": -limit, "vmax": limit}
        strong = np.abs(values) > 0.6 * limit
        cell = "{:+.1f}"
    else:
        names = ["baseline", *variants]
        values = (table * 100).to_numpy(dtype=float)
        finite = np.isfinite(values)
        scale = {"cmap": SEQUENTIAL, "vmin": 0.0, "vmax": 100.0}
        strong = values > 60
        cell = "{:.0f}"
    figure = Figure(figsize=(COLUMN_WIDTH * 1.35, 0.2 * len(names) + 0.9), layout="constrained")
    axis = figure.subplots()
    image = axis.imshow(values, aspect="auto", **scale)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if finite[i, j]:
                axis.text(
                    j,
                    i,
                    cell.format(values[i, j]),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=SURFACE if strong[i, j] else "black",
                )
    axis.set_xticks(range(len(methods)), methods)
    axis.set_yticks(range(len(names)), names)
    axis.set_xticks(np.arange(-0.5, len(methods)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(names)), minor=True)
    axis.grid(visible=False, which="major")
    axis.grid(visible=True, which="minor", color=SURFACE, linewidth=1.5)
    axis.tick_params(which="both", length=0)
    figure.colorbar(image, ax=axis, shrink=0.9, label=label)
    save(figure, path)
