"""Stage 5: all figures from saved artefacts (headless matplotlib → reports/figures)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # must run before sharded_index.plots imports pyplot

import pandas as pd

from sharded_index import TermPartition
from sharded_index.clusters import build_term_graph
from sharded_index.config import (
    EVAL_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PARTITIONS_DIR,
    load_params,
    strategy_names,
)
from sharded_index.plots import (
    plot_cluster_size_distribution,
    plot_cluster_wordclouds,
    plot_cover_fanout_by_split,
    plot_fanout_distribution,
    plot_graph_clusters,
    plot_inter_cluster_heatmap,
    plot_recall_vs_fanout,
    plot_replication_tradeoff,
    plot_tsne_projection,
)


def label(strategy: str) -> str:
    """hash → Hash, leiden_r2 → Leiden ×2, leiden_bal_r2 → Leiden-bal ×2."""
    fixed = {
        "hash": "Hash", "leiden": "Leiden",
        "leiden_aff": "Leiden-aff", "leiden_bal": "Leiden-bal",
    }
    if strategy in fixed:
        return fixed[strategy]
    for prefix, name in (("leiden_aff_r", "Leiden-aff ×"), ("leiden_bal_r", "Leiden-bal ×")):
        if strategy.startswith(prefix):
            return name + strategy.removeprefix(prefix)
    return strategy.replace("leiden_r", "Leiden ×")


def main() -> None:
    params = load_params()
    strategies = strategy_names(params)
    fig_params = params["figures"]
    top_n = fig_params["top_n_clusters"]
    seed = fig_params["layout_seed"]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Основные фигуры считаются по главному ярусу — holdout.
    fanouts_frame = pd.read_parquet(EVAL_DIR / "fanouts.parquet")
    holdout_frame = fanouts_frame[fanouts_frame["split"] == "holdout"]
    fanouts = {
        label(strategy): group["fanout"].tolist()
        for strategy, group in holdout_frame.groupby("strategy", sort=False)
    }
    plot_fanout_distribution(fanouts, save_path=FIGURES_DIR / "fanout_distribution.pdf")

    metrics = json.loads((METRICS_DIR / "metrics.json").read_text())
    raw_curves = json.loads((METRICS_DIR / "recall_by_fanout.json").read_text())["holdout"]
    curves = {label(s): {int(f): r for f, r in c.items()} for s, c in raw_curves.items()}
    holdout_of = {s: metrics[s]["by_split"]["holdout"] for s in strategies}
    mean_fanouts = {label(s): holdout_of[s]["cover_fanout"]["mean"] for s in strategies}
    plot_recall_vs_fanout(curves, mean_fanouts, save_path=FIGURES_DIR / "recall_vs_fanout.pdf")

    tradeoff = {
        label(s): {
            "duplication": metrics[s]["duplication"],
            "cover_fanout_mean": holdout_of[s]["cover_fanout"]["mean"],
            "recall_at_1": holdout_of[s]["recall_at"]["1"],
        }
        for s in strategies
    }
    plot_replication_tradeoff(tradeoff, save_path=FIGURES_DIR / "duplication_vs_fanout.pdf")

    by_split = {
        label(s): {
            split: entry["cover_fanout"]["mean"]
            for split, entry in metrics[s]["by_split"].items()
        }
        for s in strategies
    }
    plot_cover_fanout_by_split(by_split, save_path=FIGURES_DIR / "fanout_by_split.pdf")

    core = TermPartition.load(PARTITIONS_DIR / "leiden_core")
    plot_cluster_size_distribution(core.clusters_df,
                                   save_path=FIGURES_DIR / "cluster_size_distribution.pdf")
    plot_cluster_wordclouds(core, top_n=top_n,
                            save_path=FIGURES_DIR / "word_clouds_top25.pdf")

    graph = build_term_graph(core)
    plot_graph_clusters(graph, core, max_nodes=fig_params["graph_max_nodes"], seed=seed,
                        save_path=FIGURES_DIR / "npmi_graph_clusters.pdf")
    plot_tsne_projection(graph, core, top_n_clusters=top_n, seed=seed,
                         save_path=FIGURES_DIR / "tsne_2d_projection.pdf")
    plot_inter_cluster_heatmap(core, top_n_clusters=top_n,
                               save_path=FIGURES_DIR / "inter_cluster_heatmap.pdf")
    print(f"9 figures -> {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
