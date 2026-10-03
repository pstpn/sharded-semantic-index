"""Figures of one clustering method: its benchmark results and its cluster structure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from wordcloud import WordCloud

from sharded_index.config import Strategy
from sharded_index.evaluation.measure import FULL
from sharded_index.reporting.labels import Labels
from sharded_index.reporting.style import (
    BLUE,
    COLUMN_WIDTH,
    INK_MUTED,
    PAGE_WIDTH,
    SEQUENTIAL,
    SURFACE,
    categorical_x,
    marker_style,
    method_colors,
    method_markers,
    save,
    strategy_color,
    strategy_line_style,
    strategy_marker,
    themed,
)

LABEL_OFFSETS = [
    (8, 5),
    (8, -16),
    (-8, 5),
    (-8, -16),
    (8, 26),
    (-8, 26),
    (8, -37),
    (-8, -37),
    (30, 48),
    (-30, 48),
    (30, -59),
    (-30, -59),
    (50, 5),
    (-50, 5),
    (50, -16),
    (-50, -16),
]
"""Candidate label positions in points from a marker; the first free of overlaps is used."""


def _cluster_colors(clusters: Sequence[int]) -> dict[int, tuple[float, ...]]:
    unique = sorted(set(clusters))
    palette = mpl.colormaps["tab20"].resampled(max(len(unique), 1))
    return {cluster: palette(i) for i, cluster in enumerate(unique)}


def _largest_clusters(clustering: Mapping[str, int], top_n: int) -> list[int]:
    sizes = pd.Series(clustering).value_counts()
    return sizes.head(top_n).index.tolist()


def term_graph(edges: pd.DataFrame) -> nx.Graph:
    """NetworkX view of the edge list."""
    return nx.from_pandas_edgelist(edges, source="src", target="dst", edge_attr=["weight", "count"])


@themed
def plot_fanout_ecdf(
    fanouts: Mapping[str, np.ndarray], text: Labels, title: str, path: Path
) -> None:
    """Empirical CDF of the cover size per strategy; the value at 1 is the single-shard share."""
    figure = Figure(figsize=(PAGE_WIDTH, 2.8), layout="constrained")
    axis = figure.subplots()
    upper = 1
    for name, values in fanouts.items():
        ordered = np.sort(np.asarray(values))
        share = np.arange(1, len(ordered) + 1) / len(ordered)
        axis.step(
            ordered,
            share,
            where="post",
            color=strategy_color(name),
            linestyle=strategy_line_style(name),
            label=f"{name} ({text('mean')} {ordered.mean():.2f})",
        )
        upper = max(upper, int(np.percentile(ordered, 99)))
    axis.set_xlim(0.5, upper + 0.5)
    axis.set_ylim(0, 1.02)
    axis.set_xticks(range(1, upper + 1))
    categorical_x(axis)
    axis.set_xlabel(text("fanout"))
    axis.set_ylabel(text("share_of_queries"))
    axis.set_title(title, loc="left")
    axis.legend(loc="lower right", ncol=2)
    save(figure, path)


@themed
def plot_overlap_by_budget(
    curves: Mapping[str, Mapping[str, float]],
    top_k: int,
    text: Labels,
    title: str,
    path: Path,
) -> None:
    """Overlap with the unsharded top-k as a function of the shard budget."""
    figure = Figure(figsize=(COLUMN_WIDTH * 1.5, 2.8), layout="constrained")
    axis = figure.subplots()
    budgets = list(next(iter(curves.values())))
    positions = list(range(len(budgets)))
    lowest = 1.0
    for name, curve in curves.items():
        color = strategy_color(name)
        values = [curve[budget] for budget in budgets]
        lowest = min(lowest, *values)
        axis.plot(
            positions,
            values,
            color=color,
            linestyle=strategy_line_style(name),
            label=name,
            **marker_style(name, color),
        )
    axis.set_xticks(positions, [text("full_cover") if b == FULL else b for b in budgets])
    categorical_x(axis)
    axis.set_xlabel(text("shards_probed"))
    axis.set_ylabel(text("overlap_at_budget", k=top_k))
    axis.set_ylim(max(0.0, lowest - 0.05), 1.01)
    axis.set_title(title, loc="left")
    axis.legend(loc="lower right", ncol=2)
    save(figure, path)


def _label_points(
    axis: Any,
    figure: Figure,
    points: Sequence[tuple[float, float]],
    labels: Sequence[str],
    colors: Sequence[Any],
) -> list[Bbox]:
    """Annotate the points so that no label covers another label or a marker.

    A label tries ``LABEL_OFFSETS`` in order and takes the first position that
    lies inside the axes and overlaps nothing; a label moved away from the
    default position gets a leader line to its marker.  Returns the bounding
    boxes of the labels in pixels.
    """
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    inside = axis.bbox
    markers = [
        Bbox.from_bounds(px - 6, py - 6, 12, 12) for px, py in axis.transData.transform(points)
    ]
    placed: list[Bbox] = []

    def fits(extent: Bbox) -> bool:
        return bool(
            inside.x0 <= extent.x0 <= extent.x1 <= inside.x1
            and inside.y0 <= extent.y0 <= extent.y1 <= inside.y1
        )

    def annotate(index: int, offset: tuple[int, int], leader: bool) -> Any:
        return axis.annotate(
            labels[index],
            points[index],
            textcoords="offset points",
            xytext=offset,
            fontsize=6.5,
            ha="left" if offset[0] > 0 else "right",
            va="bottom" if offset[1] > 0 else "top",
            arrowprops={"arrowstyle": "-", "color": colors[index], "linewidth": 0.5}
            if leader
            else None,
        )

    for i in range(len(points)):
        obstacles = placed + markers[:i] + markers[i + 1 :]
        for offset in LABEL_OFFSETS:
            annotation = annotate(i, offset, leader=False)
            extent = annotation.get_window_extent(renderer)
            box = extent.expanded(1.08, 1.15)
            if fits(extent) and not any(box.overlaps(other) for other in obstacles):
                break
            annotation.remove()
        else:
            offset = LABEL_OFFSETS[0]
            annotation = annotate(i, offset, leader=False)
            box = annotation.get_window_extent(renderer)
        if offset != LABEL_OFFSETS[0]:
            annotation.remove()
            annotate(i, offset, leader=True)
        placed.append(box)
    return placed


@themed
def plot_duplication_vs_fanout(points: pd.DataFrame, text: Labels, title: str, path: Path) -> None:
    """Storage cost against routing cost, one point per strategy.

    ``points`` has columns ``strategy``, ``duplication``, ``fanout_mean`` and ``label``.
    Every point carries its label, placed where it covers nothing else.
    """
    figure = Figure(figsize=(PAGE_WIDTH, 3.2), layout="constrained")
    axis = figure.subplots()
    colors = [strategy_color(name) for name in points["strategy"]]
    for row, color in zip(points.itertuples(index=False), colors, strict=True):
        hollow = Strategy.parse(row.strategy).is_hash
        axis.scatter(
            row.duplication,
            row.fanout_mean,
            s=36,
            marker=strategy_marker(row.strategy),
            facecolors=SURFACE if hollow else color,
            edgecolors=color,
            linewidths=0.9,
            zorder=3,
        )
    axis.axhline(y=1.0, color=INK_MUTED, linestyle="--", linewidth=0.6)
    axis.annotate(
        text("one_shard_line"),
        (0, 1.0),
        xytext=(4, 3),
        textcoords="offset points",
        fontsize=6.5,
        color=INK_MUTED,
    )
    bottom, top = 0.9, float(points["fanout_mean"].max())
    axis.set_xlim(left=0, right=float(points["duplication"].max()) * 1.15)
    axis.set_ylim(bottom=bottom, top=top + 0.25 * (top - bottom))
    axis.set_xlabel(text("duplication"))
    axis.set_ylabel(text("fanout_mean"))
    axis.set_title(title, loc="left")
    _label_points(
        axis,
        figure,
        list(zip(points["duplication"], points["fanout_mean"], strict=True)),
        list(points["label"]),
        colors,
    )
    save(figure, path)


@themed
def plot_fanout_by_sample(fanout: pd.DataFrame, text: Labels, title: str, path: Path) -> None:
    """Mean cover size of every strategy (rows) on every query sample (columns).

    Within a sample the strategies are spread side by side in table order.
    """
    strategies = list(fanout.index)
    samples = list(fanout.columns)
    figure = Figure(figsize=(PAGE_WIDTH, 3.0), layout="constrained")
    axis = figure.subplots()
    x = np.arange(len(samples))
    dodge = 0.72 / max(len(strategies), 1)
    for i, name in enumerate(strategies):
        color = strategy_color(name)
        axis.plot(
            x + (i - (len(strategies) - 1) / 2) * dodge,
            fanout.loc[name],
            linestyle="",
            color=color,
            label=name,
            **marker_style(name, color),
        )
    for boundary in range(1, len(samples)):
        axis.axvline(boundary - 0.5, color="#dddddd", linewidth=0.5, zorder=0)
    axis.set_xticks(x, [text.sample(sample) for sample in samples])
    categorical_x(axis)
    axis.set_xlim(-0.5, len(samples) - 0.5)
    axis.set_ylim(bottom=0.9)
    axis.set_ylabel(text("fanout_mean"))
    axis.set_title(title, loc="left")
    figure.legend(loc="outside upper center", ncol=5)
    save(figure, path)


@themed
def plot_methods_comparison(
    retrieval: pd.DataFrame,
    top_k: int,
    budgets: Sequence[int],
    text: Labels,
    title: str,
    path: Path,
) -> None:
    """Overlap of every strategy under every method with confidence intervals, a panel per budget.

    ``retrieval`` holds the rows of one sample of the retrieval table.
    """
    methods = list(dict.fromkeys(retrieval["method"]))
    strategies = list(dict.fromkeys(retrieval["strategy"]))
    colors, markers = method_colors(methods), method_markers(methods)
    figure = Figure(figsize=(PAGE_WIDTH, 2.1 * len(budgets) + 0.6), layout="constrained")
    axes = figure.subplots(len(budgets), 1, sharex=True, squeeze=False)[:, 0]
    x = np.arange(len(strategies))
    dodge = 0.7 / max(len(methods), 1)
    hollow = np.array([Strategy.parse(name).is_hash for name in strategies])
    for index, (axis, budget) in enumerate(zip(axes, budgets, strict=True)):
        part = retrieval[retrieval["budget"] == str(budget)]
        for k, method in enumerate(methods):
            rows = part[part["method"] == method].set_index("strategy").reindex(strategies)
            values = rows["overlap"].to_numpy(dtype=float)
            low = rows["overlap_ci_low"].to_numpy(dtype=float)
            high = rows["overlap_ci_high"].to_numpy(dtype=float)
            offset = (k - (len(methods) - 1) / 2) * dodge
            for mask, face in ((~hollow, colors[method]), (hollow, SURFACE)):
                if mask.any():
                    axis.errorbar(
                        x[mask] + offset,
                        values[mask],
                        yerr=[values[mask] - low[mask], high[mask] - values[mask]],
                        marker=markers[method],
                        color=colors[method],
                        markerfacecolor=face,
                        elinewidth=0.6,
                        capsize=0,
                        linestyle="",
                        markersize=3.5,
                    )
        for i in range(1, len(strategies)):
            if Strategy.parse(strategies[i]).family != Strategy.parse(strategies[i - 1]).family:
                axis.axvline(i - 0.5, color="#dddddd", linewidth=0.5, zorder=0)
        axis.set_ylabel(text("overlap_at", k=top_k, shards=text.shards(budget)))
        axis.set_title(
            f"{text.panel(index)} {text('probing', shards=text.shards(budget))}", loc="left"
        )
    axes[-1].set_xticks(x, strategies)
    categorical_x(axes[-1])
    axes[-1].set_xlim(-0.6, len(strategies) - 0.4)
    figure.suptitle(title)
    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker=markers[name],
            color=color,
            markerfacecolor=color,
            label=name,
        )
        for name, color in colors.items()
    ] + [
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
            color=INK_MUTED,
            markerfacecolor=SURFACE,
            label=text("hash_baseline"),
        ),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    save(figure, path)


@themed
def plot_cluster_sizes(
    clustering: Mapping[str, int], top_n: int, text: Labels, title: str, path: Path
) -> None:
    """Histogram of cluster sizes and the sizes of the largest clusters."""
    sizes = pd.Series(clustering).value_counts()
    figure = Figure(figsize=(PAGE_WIDTH, 2.4), layout="constrained")
    histogram, largest = figure.subplots(1, 2)
    histogram.hist(sizes.to_numpy(), bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
    histogram.set_xlabel(text("cluster_size"))
    histogram.set_ylabel(text("clusters"))
    histogram.set_yscale("log")
    histogram.set_title(title, loc="left")

    top = sizes.head(top_n)
    positions = range(len(top))
    largest.barh(positions, top.to_numpy(), color=BLUE, linewidth=0)
    largest.set_yticks(positions, [str(cluster) for cluster in top.index])
    largest.tick_params(axis="y", which="minor", left=False, right=False)
    largest.set_xlabel(text("cluster_size"))
    largest.set_ylabel(text("cluster"))
    largest.set_title(text("largest_clusters", n=len(top)), loc="left")
    largest.invert_yaxis()
    save(figure, path)


@themed
def plot_cluster_wordclouds(
    clustering: Mapping[str, int],
    strength: Mapping[str, float],
    top_n: int,
    seed: int,
    text: Labels,
    title: str,
    path: Path,
) -> None:
    """Word cloud of each of the largest clusters; word size follows node strength."""
    clusters = _largest_clusters(clustering, top_n)
    n_columns = 5
    n_rows = -(-len(clusters) // n_columns)
    figure = Figure(figsize=(PAGE_WIDTH, 1.1 * n_rows + 0.3), layout="constrained")
    axes = figure.subplots(n_rows, n_columns, squeeze=False).ravel()
    for axis in axes:
        axis.axis("off")
    for axis, cluster in zip(axes, clusters, strict=False):
        words = {
            term: max(strength.get(term, 0.01), 0.01)
            for term, term_cluster in clustering.items()
            if term_cluster == cluster
        }
        cloud = WordCloud(
            width=400,
            height=300,
            background_color="white",
            max_words=50,
            colormap="viridis",
            prefer_horizontal=0.7,
            random_state=seed,
        ).generate_from_frequencies(words)
        axis.imshow(cloud, interpolation="bilinear")
        axis.set_title(text("cluster_terms", c=cluster, n=len(words)), fontsize=6)
    figure.suptitle(title)
    save(figure, path)


@themed
def plot_graph_clusters(
    edges: pd.DataFrame,
    clustering: Mapping[str, int],
    max_nodes: int,
    seed: int,
    text: Labels,
    title: str,
    path: Path,
    n_labels: int = 30,
) -> None:
    """Spring layout of the highest-degree part of the graph, nodes coloured by cluster."""
    graph = term_graph(edges)
    if graph.number_of_nodes() > max_nodes:
        top = sorted(graph.nodes(), key=lambda node: (-graph.degree(node), node))[:max_nodes]
        graph = graph.subgraph(top).copy()
    nodes = sorted(graph.nodes())
    colors = _cluster_colors([clustering[node] for node in nodes])

    layout = nx.spring_layout(graph, k=0.5, iterations=80, seed=seed)
    figure = Figure(figsize=(PAGE_WIDTH, PAGE_WIDTH), layout="constrained")
    axis = figure.subplots()
    nx.draw_networkx_edges(graph, layout, alpha=0.15, edge_color=INK_MUTED, width=0.4, ax=axis)
    nx.draw_networkx_nodes(
        graph,
        layout,
        nodelist=nodes,
        node_color=[colors[clustering[node]] for node in nodes],
        node_size=[1.5 * (1 + graph.degree(node)) for node in nodes],
        alpha=0.85,
        linewidths=0,
        ax=axis,
    )
    labelled = sorted(nodes, key=lambda node: (-graph.degree(node), node))[:n_labels]
    nx.draw_networkx_labels(graph, layout, {node: node for node in labelled}, font_size=6, ax=axis)
    axis.set_title(f"{title} ({text('terms_count', n=graph.number_of_nodes())})", loc="left")
    axis.axis("off")
    save(figure, path)


@themed
def plot_tsne(
    edges: pd.DataFrame,
    clustering: Mapping[str, int],
    strength: Mapping[str, float],
    top_n: int,
    seed: int,
    text: Labels,
    title: str,
    path: Path,
    n_labels: int = 25,
) -> None:
    """t-SNE of the adjacency rows of the terms of the largest clusters (cosine metric)."""
    graph = term_graph(edges)
    kept = set(_largest_clusters(clustering, top_n))
    nodes = sorted(node for node in graph.nodes() if clustering[node] in kept)
    adjacency = nx.to_scipy_sparse_array(
        graph.subgraph(nodes), nodelist=nodes, weight="weight", format="csr"
    )
    features = normalize(adjacency.astype(np.float32), norm="l2", axis=1)
    points = TSNE(
        n_components=2,
        perplexity=min(30, len(nodes) - 1),
        metric="cosine",
        init="random",
        random_state=seed,
    ).fit_transform(features.toarray())

    clusters = [clustering[node] for node in nodes]
    colors = _cluster_colors(clusters)
    strengths = np.array([strength.get(node, 1.0) for node in nodes])
    figure = Figure(figsize=(PAGE_WIDTH, PAGE_WIDTH * 0.8), layout="constrained")
    axis = figure.subplots()
    axis.scatter(
        points[:, 0],
        points[:, 1],
        c=[colors[cluster] for cluster in clusters],
        s=np.clip(2 + strengths * 2, 3, 100),
        alpha=0.8,
        edgecolors=SURFACE,
        linewidths=0.2,
    )
    for i in np.argsort(strengths, kind="stable")[-n_labels:]:
        axis.annotate(nodes[i], (points[i, 0], points[i, 1]), fontsize=6)
    axis.set_title(f"{title} ({text('clusters_terms', c=len(colors), n=len(nodes))})", loc="left")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.axis("equal")
    save(figure, path)


@themed
def plot_cluster_heatmap(
    edges: pd.DataFrame,
    clustering: Mapping[str, int],
    top_n: int,
    text: Labels,
    title: str,
    path: Path,
) -> None:
    """Total edge weight between the largest clusters (diagonal: inside a cluster)."""
    clusters = _largest_clusters(clustering, top_n)
    src = edges["src"].map(clustering)
    dst = edges["dst"].map(clustering)
    among = src.isin(clusters) & dst.isin(clusters)
    weights = (
        edges[among]
        .assign(src_cluster=src[among], dst_cluster=dst[among])
        .groupby(["src_cluster", "dst_cluster"])["weight"]
        .sum()
    )
    matrix = pd.DataFrame(0.0, index=clusters, columns=clusters)
    for (a, b), weight in weights.items():
        matrix.loc[a, b] += weight
        if a != b:
            matrix.loc[b, a] += weight

    figure = Figure(figsize=(PAGE_WIDTH, PAGE_WIDTH * 0.85), layout="constrained")
    axis = figure.subplots()
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 5},
        cmap=SEQUENTIAL,
        linewidths=0.5,
        linecolor=SURFACE,
        ax=axis,
        cbar_kws={"label": text("total_edge_weight")},
    )
    axis.set_title(title, loc="left")
    axis.set_xlabel(text("cluster"))
    axis.set_ylabel(text("cluster"))
    axis.tick_params(which="both", length=0)
    save(figure, path)
