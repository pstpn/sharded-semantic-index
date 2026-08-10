"""Figures for the benchmark and the cluster analysis.

Every function draws one figure, optionally saves it to ``save_path``
(the notebook passes paths under ``results/``) and shows it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from wordcloud import WordCloud

from .partition import TermPartition

STRATEGY_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#9b59b6",
    "#e67e22", "#16a085", "#34495e", "#c0392b",
]
STRATEGY_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

_HEADLESS_BACKENDS = {"agg", "pdf", "ps", "svg", "template", "cairo"}


def _finish(fig, save_path: Path | None) -> None:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if mpl.get_backend().lower() in _HEADLESS_BACKENDS:
        plt.close(fig)  # pipeline run: nothing to display
    else:
        plt.show()


def _cluster_colors(cluster_ids: list[int]) -> dict[int, tuple]:
    """Stable color per cluster id from the tab20 palette."""
    unique = sorted(set(cluster_ids))
    cmap = mpl.colormaps["tab20"].resampled(max(len(unique), 1))
    return {c: cmap(i) for i, c in enumerate(unique)}


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_fanout_distribution(
    fanouts: dict[str, list[int]],
    save_path: Path | None = None,
) -> None:
    """ECDF of query cover fanout, one curve per strategy.

    Overlaid histograms of seven strategies are unreadable; the ECDF keeps
    every strategy visible and reads share-covered-by-k directly off the
    y axis (x = 1 gives the share of single-shard queries).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    max_x = 0
    for (name, values), color in zip(fanouts.items(), STRATEGY_COLORS):
        sorted_vals = np.sort(np.asarray(values))
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.step(sorted_vals, ecdf, where="post", linewidth=2,
                label=f"{name} (mean={np.mean(values):.2f})", color=color)
        max_x = max(max_x, int(np.percentile(sorted_vals, 99)))
    ax.axvline(x=1, color="green", linestyle="--", alpha=0.5, label="single shard")
    ax.set_xlim(0.5, max_x + 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(1, max_x + 1))
    ax.set_xlabel("Cover Fanout (shards to cover the query)")
    ax.set_ylabel("Share of Queries (ECDF)")
    ax.set_title("Cover Fanout: Empirical CDF by Strategy")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path)


def plot_recall_vs_fanout(
    recall_curves: dict[str, dict[int, float]],
    mean_fanouts: dict[str, float] | None = None,
    save_path: Path | None = None,
) -> None:
    """Routing recall as a function of the number of top-ranked shards probed."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, curve) in enumerate(recall_curves.items()):
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        marker = STRATEGY_MARKERS[i % len(STRATEGY_MARKERS)]
        fanouts = sorted(curve)
        recalls = [curve[f] for f in fanouts]

        label = name
        if mean_fanouts and name in mean_fanouts:
            label = f"{name} (mean fanout={mean_fanouts[name]:.1f})"
        ax.plot(fanouts, recalls, marker=marker, linewidth=2, color=color, label=label)

        # Per-point value labels get unreadable with many curves — legend suffices.
        if len(recall_curves) <= 4:
            y_offset = 10 if i % 2 == 0 else -15
            for f, r in zip(fanouts, recalls):
                ax.annotate(f"{r:.2f}", (f, r), textcoords="offset points",
                            xytext=(0, y_offset), ha="center", fontsize=9, color=color)

    all_fanouts = sorted({f for curve in recall_curves.values() for f in curve})
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Full Index")
    ax.set_xlabel("Number of Shards Probed")
    ax.set_ylabel("Routing Recall")
    ax.set_title("Routing Recall vs Fanout")
    ax.set_xticks(all_fanouts)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path)


def plot_cover_fanout_by_split(
    data: dict[str, dict[str, float]],
    save_path: Path | None = None,
) -> None:
    """Grouped bars: mean cover fanout per strategy across evaluation splits.

    ``data[strategy_label][split] = mean cover fanout``.  The train/holdout
    gap shows generalization; the holdout/ood gap shows distribution transfer.
    """
    strategies = list(data)
    splits = list(dict.fromkeys(s for per_split in data.values() for s in per_split))
    width = 0.8 / max(len(splits), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(strategies))
    for j, split in enumerate(splits):
        values = [data[s].get(split, np.nan) for s in strategies]
        offset = (j - (len(splits) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width * 0.95, label=split,
                      color=STRATEGY_COLORS[j % len(STRATEGY_COLORS)], alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)

    ax.set_xticks(x, strategies, rotation=15)
    ax.set_ylabel("Mean Cover Fanout")
    ax.set_title("Cover Fanout by Evaluation Split")
    ax.legend(title="split")
    ax.grid(True, axis="y", alpha=0.3)
    _finish(fig, save_path)


def plot_replication_tradeoff(
    stats: dict[str, dict[str, float]],
    save_path: Path | None = None,
) -> None:
    """The research trade-off: duplication (storage cost) vs cover fanout.

    One point per strategy; ``stats[label]`` needs keys ``duplication``,
    ``cover_fanout_mean`` and optionally ``recall_at_1``.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, s) in enumerate(stats.items()):
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        marker = STRATEGY_MARKERS[i % len(STRATEGY_MARKERS)]
        ax.scatter(s["duplication"], s["cover_fanout_mean"], s=140,
                   color=color, marker=marker, zorder=3, label=label)
        note = f"{label}"
        if "recall_at_1" in s:
            note += f"\nrecall@1={s['recall_at_1']:.2f}"
        ax.annotate(note, (s["duplication"], s["cover_fanout_mean"]),
                    textcoords="offset points", xytext=(10, 6), fontsize=9, color=color)

    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.6,
               label="ideal: fanout = 1")
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.8,
               label="1x storage (document sharding)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.9)
    ax.set_xlabel("Document Duplication (x)")
    ax.set_ylabel("Mean Cover Fanout (shards to cover the query)")
    ax.set_title("Replication Trade-off: Duplication vs Cover Fanout")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path)


# ═══════════════════════════════════════════════════════════════════════════
# Cluster-structure figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_cluster_size_distribution(
    clusters_df: pd.DataFrame,
    top_n: int = 25,
    save_path: Path | None = None,
) -> None:
    """Histogram, boxplot and top-N bar chart of cluster sizes."""
    sizes = clusters_df["size"].values
    fig, (ax_hist, ax_box, ax_top) = plt.subplots(1, 3, figsize=(15, 4))

    ax_hist.hist(sizes, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax_hist.set_xlabel("Cluster Size (terms)")
    ax_hist.set_ylabel("Number of Clusters")
    ax_hist.set_title("Cluster Size Distribution")
    ax_hist.set_yscale("log")

    ax_box.boxplot(sizes, vert=True)
    ax_box.set_ylabel("Cluster Size")
    ax_box.set_title("Boxplot of Sizes")

    top_clusters = clusters_df.head(top_n)
    ax_top.barh(top_clusters["cluster"].astype(str), top_clusters["size"], color="coral")
    ax_top.set_xlabel("Cluster Size")
    ax_top.set_ylabel("Cluster ID")
    ax_top.set_title(f"Top-{top_n} Largest Clusters")
    ax_top.invert_yaxis()

    _finish(fig, save_path)


def plot_cluster_wordclouds(
    partition: TermPartition,
    top_n: int = 25,
    max_words: int = 50,
    save_path: Path | None = None,
) -> None:
    """Word cloud per cluster for the top-N largest clusters (word size ~ strength)."""
    top_cluster_ids = partition.clusters_df["cluster"].head(top_n).tolist()

    n_cols = 5
    n_rows = -(-len(top_cluster_ids) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for ax, cluster_id in zip(np.ravel(axes), top_cluster_ids + [None] * (n_rows * n_cols)):
        ax.axis("off")
        if cluster_id is None:
            continue
        words = {
            term: max(partition.node_strength.get(term, 0.01), 0.01)
            for term, c in partition.term_to_shard.items()
            if c == cluster_id
        }
        if not words:
            ax.text(0.5, 0.5, "Empty cluster", ha="center", va="center")
            continue
        cloud = WordCloud(
            width=400, height=300,
            background_color="white",
            max_words=max_words,
            colormap="viridis",
            prefer_horizontal=0.7,
        ).generate_from_frequencies(words)
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(f"Cluster {cluster_id} (size={len(words)})", fontsize=11)

    fig.suptitle(f"Word Clouds: Top-{top_n} Clusters (word size ~ strength)", fontsize=13)
    _finish(fig, save_path)


def plot_graph_clusters(
    graph: nx.Graph,
    partition: TermPartition,
    max_nodes: int = 1000,
    node_size_factor: int = 5,
    edge_alpha: float = 0.15,
    n_labels: int = 30,
    seed: int = 42,
    save_path: Path | None = None,
) -> None:
    """Spring layout of the NPMI graph, nodes colored by Leiden cluster.

    Only the ``max_nodes`` highest-degree nodes are drawn; the ``n_labels``
    highest-degree ones get text labels.
    """
    if graph.number_of_nodes() > max_nodes:
        top_nodes = sorted(graph.nodes(), key=graph.degree, reverse=True)[:max_nodes]
        subgraph = graph.subgraph(top_nodes).copy()
        print(f"Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    else:
        subgraph = graph

    term_to_shard = partition.term_to_shard
    colors = _cluster_colors([term_to_shard.get(n, -1) for n in subgraph.nodes()])
    node_colors = [
        colors.get(term_to_shard.get(n, -1), (0.5, 0.5, 0.5, 1.0)) for n in subgraph.nodes()
    ]
    node_sizes = [node_size_factor * (1 + subgraph.degree(n)) for n in subgraph.nodes()]

    pos = nx.spring_layout(subgraph, k=0.5, iterations=80, seed=seed)
    fig, ax = plt.subplots(figsize=(20, 20))
    nx.draw_networkx_edges(subgraph, pos, alpha=edge_alpha, edge_color="gray", ax=ax)
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=node_sizes,
                           ax=ax, alpha=0.85)

    labeled = sorted(subgraph.nodes(), key=subgraph.degree, reverse=True)[:n_labels]
    nx.draw_networkx_labels(subgraph, pos, {n: n for n in labeled}, font_size=8, ax=ax)

    ax.set_title(f"NPMI Graph with Leiden Clusters ({subgraph.number_of_nodes()} nodes)",
                 fontsize=12)
    ax.axis("off")
    _finish(fig, save_path)


def plot_tsne_projection(
    graph: nx.Graph,
    partition: TermPartition,
    top_n_clusters: int = 25,
    n_labels: int = 25,
    seed: int = 42,
    save_path: Path | None = None,
) -> None:
    """t-SNE of graph adjacency rows for terms of the top-N clusters.

    Each term is represented by its L2-normalized adjacency row (cosine
    metric), so terms sharing neighbors land close together.
    """
    term_to_shard = partition.term_to_shard
    cluster_sizes = pd.Series(term_to_shard).value_counts()
    top_clusters = set(cluster_sizes.head(top_n_clusters).index)
    nodes = [n for n in graph.nodes() if term_to_shard.get(n, -1) in top_clusters]
    print(f"Selected {len(nodes)} nodes from top-{top_n_clusters} clusters")

    subgraph = graph.subgraph(nodes)
    adjacency = nx.to_scipy_sparse_array(subgraph, nodelist=nodes, weight="weight", format="csr")
    features = normalize(adjacency.astype(np.float32), norm="l2", axis=1)

    coords = TSNE(
        n_components=2,
        perplexity=min(30, len(nodes) - 1),
        metric="cosine",
        init="random",
        random_state=seed,
        n_jobs=-1,
    ).fit_transform(features.toarray())

    clusters = np.array([term_to_shard.get(n, -1) for n in nodes])
    strengths = np.array([partition.node_strength.get(n, 1.0) for n in nodes])
    colors_by_id = _cluster_colors(list(clusters))
    point_colors = [colors_by_id[c] for c in clusters]
    point_sizes = np.clip(5 + strengths * 5, 10, 300)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=point_sizes,
               alpha=0.8, edgecolors="white", linewidths=0.3)

    for idx in np.argsort(strengths)[-n_labels:]:
        ax.annotate(nodes[idx], (coords[idx, 0], coords[idx, 1]), fontsize=7)

    ax.set_title(f"2D Projection (t-SNE, {len(colors_by_id)} clusters, {len(nodes)} nodes)",
                 fontsize=12)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.axis("equal")
    _finish(fig, save_path)


def plot_inter_cluster_heatmap(
    partition: TermPartition,
    top_n_clusters: int = 25,
    save_path: Path | None = None,
) -> None:
    """Heatmap of total NPMI weight between the top-N clusters.

    The diagonal shows intra-cluster weight; a good partition concentrates
    weight on the diagonal.
    """
    term_to_shard = partition.term_to_shard
    edges_df = partition.edges_df

    cluster_sizes = pd.Series(term_to_shard).value_counts()
    top_clusters = cluster_sizes.head(top_n_clusters).index.tolist()

    src_cluster = edges_df["src"].map(term_to_shard)
    dst_cluster = edges_df["dst"].map(term_to_shard)
    mask = src_cluster.isin(top_clusters) & dst_cluster.isin(top_clusters)

    weights = (
        edges_df[mask]
        .assign(src_c=src_cluster[mask], dst_c=dst_cluster[mask])
        .groupby(["src_c", "dst_c"])["weight"]
        .sum()
        .reset_index()
    )

    matrix = pd.DataFrame(0.0, index=top_clusters, columns=top_clusters)
    for row in weights.itertuples(index=False):
        matrix.loc[row.src_c, row.dst_c] += row.weight
        if row.src_c != row.dst_c:
            matrix.loc[row.dst_c, row.src_c] += row.weight

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Sum of NPMI Weights"},
    )
    ax.set_title(f"Inter-Cluster Connections (Top-{top_n_clusters}, sum of NPMI weights)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cluster")
    _finish(fig, save_path)
