"""Figures of one clustering method: its cluster structure and its storage/routing trade-off.

The figures carry no titles: which method and sample a figure shows is said
by its file name and by the caption that embeds it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from wordcloud import WordCloud

from sharded_index.config import Strategy
from sharded_index.reporting.labels import Labels
from sharded_index.reporting.style import (
    BLUE,
    INK_MUTED,
    PAGE_WIDTH,
    SEQUENTIAL,
    SURFACE,
    save,
    strategy_color,
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
def plot_duplication_vs_fanout(points: pd.DataFrame, text: Labels, path: Path) -> None:
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
    _label_points(
        axis,
        figure,
        list(zip(points["duplication"], points["fanout_mean"], strict=True)),
        list(points["label"]),
        colors,
    )
    save(figure, path)


@themed
def plot_cluster_sizes(clustering: Mapping[str, int], top_n: int, text: Labels, path: Path) -> None:
    """Histogram of cluster sizes (a) and the sizes of the largest clusters (b)."""
    sizes = pd.Series(clustering).value_counts()
    figure = Figure(figsize=(PAGE_WIDTH, 2.4), layout="constrained")
    histogram, largest = figure.subplots(1, 2)
    histogram.hist(sizes.to_numpy(), bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
    histogram.set_xlabel(text("cluster_size"))
    histogram.set_ylabel(text("clusters"))
    histogram.set_yscale("log")
    histogram.set_title(text.panel(0), loc="left")

    top = sizes.head(top_n)
    positions = range(len(top))
    largest.barh(positions, top.to_numpy(), color=BLUE, linewidth=0)
    largest.set_yticks(positions, [str(cluster) for cluster in top.index])
    largest.tick_params(axis="y", which="minor", left=False, right=False)
    largest.set_xlabel(text("cluster_size"))
    largest.set_ylabel(text("cluster"))
    largest.set_title(text.panel(1), loc="left")
    largest.invert_yaxis()
    save(figure, path)


@themed
def plot_cluster_wordclouds(
    clustering: Mapping[str, int],
    strength: Mapping[str, float],
    top_n: int,
    seed: int,
    text: Labels,
    path: Path,
) -> None:
    """Word cloud of each of the largest clusters; word size follows node strength."""
    clusters = _largest_clusters(clustering, top_n)
    n_columns = 5
    n_rows = -(-len(clusters) // n_columns)
    figure = Figure(figsize=(PAGE_WIDTH, 1.1 * n_rows + 0.1), layout="constrained")
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
            width=800,
            height=600,
            background_color="white",
            max_words=50,
            colormap="viridis",
            prefer_horizontal=0.7,
            random_state=seed,
        ).generate_from_frequencies(words)
        axis.imshow(cloud, interpolation="bilinear")
        axis.set_title(text("cluster_terms", c=cluster, n=len(words)), fontsize=6)
    save(figure, path)


@themed
def plot_graph_clusters(
    edges: pd.DataFrame,
    clustering: Mapping[str, int],
    max_nodes: int,
    seed: int,
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
    axis.axis("off")
    save(figure, path)


@themed
def plot_cluster_heatmap(
    edges: pd.DataFrame, clustering: Mapping[str, int], top_n: int, text: Labels, path: Path
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

    figure = Figure(figsize=(PAGE_WIDTH, PAGE_WIDTH), layout="constrained")
    axis = figure.subplots()
    axis.grid(visible=False)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 6.5},
        cmap=SEQUENTIAL,
        linewidths=0.5,
        linecolor=SURFACE,
        ax=axis,
        cbar_kws={"label": text("total_edge_weight")},
    )
    axis.set_xlabel(text("cluster"))
    axis.set_ylabel(text("cluster"))
    axis.tick_params(which="both", length=0)
    save(figure, path)
