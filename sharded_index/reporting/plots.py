"""Figures of the benchmark and of the cluster structure, written as reproducible PDFs."""

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
from matplotlib.transforms import Bbox
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from wordcloud import WordCloud

from sharded_index.evaluation.measure import FULL

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
LABEL_OFFSETS = [
    (10, 6),
    (10, -20),
    (-10, 6),
    (-10, -20),
    (10, 32),
    (-10, 32),
    (10, -46),
    (-10, -46),
    (36, 60),
    (-36, 60),
    (36, -74),
    (-36, -74),
    (60, 6),
    (-60, 6),
    (60, -20),
    (-60, -20),
]
"""Candidate label positions in points from a marker; the first free of overlaps is used."""


def _save(figure: Figure, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})


def _color(index: int) -> tuple[float, ...]:
    return mpl.colormaps["tab10"](index % 10)


def _line_style(name: str) -> str:
    return "--" if name.startswith("hash_") else "-"


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


def plot_fanout_ecdf(fanouts: Mapping[str, np.ndarray], title: str, path: Path) -> None:
    """Empirical CDF of the cover size per strategy; the value at 1 is the single-shard share."""
    figure = Figure(figsize=(10, 6), layout="tight")
    axis = figure.subplots()
    upper = 1
    for i, (name, values) in enumerate(fanouts.items()):
        ordered = np.sort(np.asarray(values))
        share = np.arange(1, len(ordered) + 1) / len(ordered)
        axis.step(
            ordered,
            share,
            where="post",
            linewidth=2,
            color=_color(i),
            linestyle=_line_style(name),
            label=f"{name} (mean {ordered.mean():.2f})",
        )
        upper = max(upper, int(np.percentile(ordered, 99)))
    axis.set_xlim(0.5, upper + 0.5)
    axis.set_ylim(0, 1.02)
    axis.set_xticks(range(1, upper + 1))
    axis.set_xlabel("Shards in the query cover")
    axis.set_ylabel("Share of queries")
    axis.set_title(title)
    axis.legend(loc="lower right")
    axis.grid(visible=True, alpha=0.3)
    _save(figure, path)


def plot_overlap_by_budget(
    curves: Mapping[str, Mapping[str, float]],
    top_k: int,
    title: str,
    path: Path,
) -> None:
    """Overlap with the unsharded top-k as a function of the shard budget."""
    figure = Figure(figsize=(10, 6), layout="tight")
    axis = figure.subplots()
    budgets = list(next(iter(curves.values())))
    positions = list(range(len(budgets)))
    for i, (name, curve) in enumerate(curves.items()):
        axis.plot(
            positions,
            [curve[budget] for budget in budgets],
            linewidth=2,
            marker=MARKERS[i % len(MARKERS)],
            color=_color(i),
            linestyle=_line_style(name),
            label=name,
        )
    axis.set_xticks(positions, ["full cover" if budget == FULL else budget for budget in budgets])
    axis.set_xlabel("Shards probed")
    axis.set_ylabel(f"Overlap@{top_k} with the unsharded index")
    axis.set_ylim(0, 1.05)
    axis.set_title(title)
    axis.legend(loc="lower right")
    axis.grid(visible=True, alpha=0.3)
    _save(figure, path)


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
        Bbox.from_bounds(px - 9, py - 9, 18, 18) for px, py in axis.transData.transform(points)
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
            fontsize=9,
            color=colors[index],
            ha="left" if offset[0] > 0 else "right",
            va="bottom" if offset[1] > 0 else "top",
            arrowprops={"arrowstyle": "-", "color": colors[index], "alpha": 0.6, "linewidth": 0.7}
            if leader
            else None,
        )

    for i in range(len(points)):
        obstacles = placed + markers[:i] + markers[i + 1 :]
        for offset in LABEL_OFFSETS:
            text = annotate(i, offset, leader=False)
            extent = text.get_window_extent(renderer)
            box = extent.expanded(1.08, 1.15)
            if fits(extent) and not any(box.overlaps(other) for other in obstacles):
                break
            text.remove()
        else:
            offset = LABEL_OFFSETS[0]
            text = annotate(i, offset, leader=False)
            box = text.get_window_extent(renderer)
        if offset != LABEL_OFFSETS[0]:
            text.remove()
            annotate(i, offset, leader=True)
        placed.append(box)
    return placed


def plot_duplication_vs_fanout(points: pd.DataFrame, title: str, path: Path) -> None:
    """Storage cost against routing cost, one point per strategy.

    ``points`` has columns ``strategy``, ``duplication``, ``fanout_mean`` and ``overlap``.
    Every point carries its own label, placed where it covers nothing else.
    """
    figure = Figure(figsize=(10, 6), layout="tight")
    axis = figure.subplots()
    for i, row in enumerate(points.itertuples(index=False)):
        axis.scatter(
            row.duplication,
            row.fanout_mean,
            s=140,
            color=_color(i),
            marker=MARKERS[i % len(MARKERS)],
            zorder=3,
        )
    axis.axhline(y=1.0, color="green", linestyle="--", alpha=0.6)
    bottom, top = 0.9, float(points["fanout_mean"].max())
    axis.set_xlim(left=0, right=float(points["duplication"].max()) * 1.15)
    axis.set_ylim(bottom=bottom, top=top + 0.25 * (top - bottom))
    axis.set_xlabel("Document duplication (shards per document)")
    axis.set_ylabel("Mean shards in the query cover")
    axis.set_title(title)
    axis.grid(visible=True, alpha=0.3)
    _label_points(
        axis,
        figure,
        list(zip(points["duplication"], points["fanout_mean"], strict=True)),
        [f"{row.strategy}\noverlap {row.overlap:.2f}" for row in points.itertuples(index=False)],
        [_color(i) for i in range(len(points))],
    )
    _save(figure, path)


def _grouped_bars(
    axis: Any, table: pd.DataFrame, errors: Mapping[str, Any] | None, fmt: str
) -> None:
    """Bars of every column of ``table`` grouped by its rows."""
    width = 0.8 / max(len(table.columns), 1)
    x = np.arange(len(table))
    for j, column in enumerate(table.columns):
        offset = (j - (len(table.columns) - 1) / 2) * width
        bars = axis.bar(
            x + offset,
            table[column],
            width * 0.95,
            yerr=None if errors is None else errors[column],
            capsize=3,
            label=column,
            color=_color(j),
            alpha=0.85,
        )
        axis.bar_label(bars, fmt=fmt, fontsize=7, padding=3)
    axis.set_xticks(x, table.index)
    axis.grid(visible=True, axis="y", alpha=0.3)


def plot_fanout_by_sample(fanout: pd.DataFrame, title: str, path: Path) -> None:
    """Mean cover size of every strategy (rows) on every query sample (columns)."""
    figure = Figure(figsize=(12, 6), layout="tight")
    axis = figure.subplots()
    _grouped_bars(axis, fanout, None, "%.2f")
    axis.tick_params(axis="x", rotation=15)
    axis.set_ylabel("Mean shards in the query cover")
    axis.set_title(title)
    axis.legend(title="sample")
    _save(figure, path)


def plot_methods_comparison(retrieval: pd.DataFrame, title: str, path: Path) -> None:
    """Overlap of every strategy under every clustering method, with confidence intervals.

    ``retrieval`` holds the rows of one sample and one budget of the retrieval table.
    """
    methods = list(dict.fromkeys(retrieval["method"]))
    strategies = list(dict.fromkeys(retrieval["strategy"]))

    def pivot(column: str) -> pd.DataFrame:
        table = retrieval.pivot(index="strategy", columns="method", values=column)
        return table.loc[strategies, methods]

    overlap, low, high = pivot("overlap"), pivot("overlap_ci_low"), pivot("overlap_ci_high")
    errors = {
        method: [overlap[method] - low[method], high[method] - overlap[method]]
        for method in methods
    }
    figure = Figure(figsize=(max(10, 1.4 * len(strategies)), 6), layout="tight")
    axis = figure.subplots()
    _grouped_bars(axis, overlap, errors, "%.2f")
    axis.set_ylabel("Overlap with the unsharded index")
    axis.set_ylim(0, 1.08)
    axis.set_title(title)
    axis.legend(title="method", loc="lower right")
    _save(figure, path)


def plot_cluster_sizes(clustering: Mapping[str, int], top_n: int, title: str, path: Path) -> None:
    """Histogram of cluster sizes and the sizes of the largest clusters."""
    sizes = pd.Series(clustering).value_counts()
    figure = Figure(figsize=(13, 4), layout="tight")
    histogram, largest = figure.subplots(1, 2)
    histogram.hist(sizes.to_numpy(), bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    histogram.set_xlabel("Cluster size (terms)")
    histogram.set_ylabel("Clusters")
    histogram.set_yscale("log")
    histogram.set_title(title)

    top = sizes.head(top_n)
    positions = range(len(top))
    largest.barh(positions, top.to_numpy(), color="coral")
    largest.set_yticks(positions, [str(cluster) for cluster in top.index])
    largest.set_xlabel("Cluster size (terms)")
    largest.set_ylabel("Cluster")
    largest.set_title(f"{len(top)} largest clusters")
    largest.invert_yaxis()
    _save(figure, path)


def plot_cluster_wordclouds(
    clustering: Mapping[str, int],
    strength: Mapping[str, float],
    top_n: int,
    seed: int,
    title: str,
    path: Path,
) -> None:
    """Word cloud of each of the largest clusters; word size follows node strength."""
    clusters = _largest_clusters(clustering, top_n)
    n_columns = 5
    n_rows = -(-len(clusters) // n_columns)
    figure = Figure(figsize=(4 * n_columns, 4 * n_rows), layout="tight")
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
        axis.set_title(f"Cluster {cluster} ({len(words)} terms)", fontsize=11)
    figure.suptitle(title, fontsize=13)
    _save(figure, path)


def plot_graph_clusters(
    edges: pd.DataFrame,
    clustering: Mapping[str, int],
    max_nodes: int,
    seed: int,
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
    figure = Figure(figsize=(20, 20), layout="tight")
    axis = figure.subplots()
    nx.draw_networkx_edges(graph, layout, alpha=0.15, edge_color="gray", ax=axis)
    nx.draw_networkx_nodes(
        graph,
        layout,
        nodelist=nodes,
        node_color=[colors[clustering[node]] for node in nodes],
        node_size=[5 * (1 + graph.degree(node)) for node in nodes],
        alpha=0.85,
        ax=axis,
    )
    labelled = sorted(nodes, key=lambda node: (-graph.degree(node), node))[:n_labels]
    nx.draw_networkx_labels(graph, layout, {node: node for node in labelled}, font_size=8, ax=axis)
    axis.set_title(f"{title} ({graph.number_of_nodes()} terms)", fontsize=12)
    axis.axis("off")
    _save(figure, path)


def plot_tsne(
    edges: pd.DataFrame,
    clustering: Mapping[str, int],
    strength: Mapping[str, float],
    top_n: int,
    seed: int,
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
    figure = Figure(figsize=(12, 10), layout="tight")
    axis = figure.subplots()
    axis.scatter(
        points[:, 0],
        points[:, 1],
        c=[colors[cluster] for cluster in clusters],
        s=np.clip(5 + strengths * 5, 10, 300),
        alpha=0.8,
        edgecolors="white",
        linewidths=0.3,
    )
    for i in np.argsort(strengths, kind="stable")[-n_labels:]:
        axis.annotate(nodes[i], (points[i, 0], points[i, 1]), fontsize=7)
    axis.set_title(f"{title} ({len(colors)} clusters, {len(nodes)} terms)", fontsize=12)
    axis.axis("equal")
    _save(figure, path)


def plot_cluster_heatmap(
    edges: pd.DataFrame,
    clustering: Mapping[str, int],
    top_n: int,
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

    figure = Figure(figsize=(12, 10), layout="tight")
    axis = figure.subplots()
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=axis,
        cbar_kws={"label": "Total edge weight"},
    )
    axis.set_title(title)
    axis.set_xlabel("Cluster")
    axis.set_ylabel("Cluster")
    _save(figure, path)
