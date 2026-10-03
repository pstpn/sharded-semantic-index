"""Draw the figures: per-method benchmark and cluster-structure plots, then the overview figures."""

from __future__ import annotations

import logging

import pandas as pd

from sharded_index.config import Config, Strategy
from sharded_index.graph.cooccurrence import node_strength
from sharded_index.paths import Paths
from sharded_index.pipeline.common import read_clustering, setup
from sharded_index.reporting import overview, plots
from sharded_index.reporting.labels import Labels

logger = logging.getLogger(__name__)


def _benchmark_figures(
    config: Config,
    paths: Paths,
    method: str,
    text: Labels,
    partitions: pd.DataFrame,
    routing: pd.DataFrame,
    retrieval: pd.DataFrame,
) -> None:
    """Cover size, overlap by budget and the storage/routing trade-off of one method."""
    sample = config.figures.sample
    strategies = list(config.partition.strategies)
    samples = [name for name in config.queries.sample_names() if name in set(routing["sample"])]
    target = paths.figures / method
    title = f"{method}, {text.sample(sample)}"

    queries = pd.read_parquet(paths.evaluation(method) / "queries.parquet")
    measured = queries[(queries["sample"] == sample) & queries["evaluated"]]
    fanouts = {
        name: measured.loc[measured["strategy"] == name, "fanout"].to_numpy() for name in strategies
    }
    plots.plot_fanout_ecdf(fanouts, text, title, target / "fanout_ecdf.pdf")

    retrieval = retrieval[(retrieval["method"] == method) & (retrieval["sample"] == sample)]
    overlap = retrieval.pivot(index="strategy", columns="budget", values="overlap")
    budgets = list(dict.fromkeys(retrieval["budget"]))
    plots.plot_overlap_by_budget(
        {name: overlap.loc[name, budgets].to_dict() for name in strategies},
        config.evaluation.top_k,
        text,
        title,
        target / "overlap_by_budget.pdf",
    )

    routing = routing[routing["method"] == method]
    fanout = routing.pivot(index="strategy", columns="sample", values="fanout_mean")
    points = pd.DataFrame(
        {
            "strategy": strategies,
            "duplication": partitions[partitions["method"] == method]
            .set_index("strategy")
            .loc[strategies, "duplication"]
            .to_numpy(),
            "fanout_mean": fanout.loc[strategies, sample].to_numpy(),
            "overlap": overlap.loc[strategies, budgets[0]].to_numpy(),
        }
    )
    plots.plot_duplication_vs_fanout(points, text, title, target / "duplication_vs_fanout.pdf")
    plots.plot_fanout_by_sample(
        fanout.loc[strategies, samples], text, method, target / "fanout_by_sample.pdf"
    )


def _structure_figures(
    config: Config,
    paths: Paths,
    method: str,
    text: Labels,
    edges: pd.DataFrame,
    strength: dict[str, float],
) -> None:
    """Sizes, contents and mutual links of the clusters of one method."""
    settings = config.figures
    top, seed = settings.top_clusters, settings.layout_seed
    target = paths.figures / method
    clustering = read_clustering(paths, method)

    plots.plot_cluster_sizes(clustering, top, text, method, target / "cluster_sizes.pdf")
    plots.plot_cluster_wordclouds(
        clustering, strength, top, seed, text, method, target / "cluster_wordclouds.pdf"
    )
    plots.plot_graph_clusters(
        edges,
        clustering,
        settings.graph_max_nodes,
        seed,
        text,
        method,
        target / "graph_clusters.pdf",
    )
    plots.plot_tsne(edges, clustering, strength, top, seed, text, method, target / "tsne.pdf")
    plots.plot_cluster_heatmap(edges, clustering, top, text, method, target / "cluster_heatmap.pdf")


def _overview_figures(
    config: Config,
    paths: Paths,
    text: Labels,
    partitions: pd.DataFrame,
    routing: pd.DataFrame,
    retrieval: pd.DataFrame,
) -> None:
    """Every method and strategy together: samples, gains over hash, costs, budgets, sweeps."""
    settings = config.figures
    sample, top_k = settings.sample, config.evaluation.top_k
    first_budget = str(config.evaluation.budgets[0])
    strategies = list(config.partition.strategies)
    semantic = [name for name in strategies if not Strategy.parse(name).is_hash]
    samples = [name for name in config.queries.sample_names() if name in set(routing["sample"])]
    counts = routing.drop_duplicates("sample").set_index("sample")["evaluated"].to_dict()
    at_first_budget = retrieval[retrieval["budget"] == first_budget]
    comparisons = pd.read_csv(paths.metrics / "comparisons.csv")

    overview.plot_metric_by_sample(
        routing,
        "single_shard_share",
        ("single_shard_ci_low", "single_shard_ci_high"),
        strategies,
        samples,
        counts,
        text,
        text("single_shard_share"),
        paths.figures / "overview_single_shard_by_sample.pdf",
        limits=(0, 1.02),
    )
    overview.plot_metric_by_sample(
        at_first_budget,
        "overlap",
        ("overlap_ci_low", "overlap_ci_high"),
        strategies,
        samples,
        counts,
        text,
        text("overlap_at_one", k=top_k),
        paths.figures / "overview_overlap_by_sample.pdf",
    )
    overview.plot_metric_by_sample(
        routing,
        "fanout_mean",
        ("fanout_ci_low", "fanout_ci_high"),
        strategies,
        samples,
        counts,
        text,
        text("fanout_mean"),
        paths.figures / "overview_fanout_by_sample.pdf",
    )
    for metric, label, name in (
        (f"overlap_{first_budget}", text("delta_overlap", k=top_k), "overview_gain_overlap.pdf"),
        ("single_shard_share", text("delta_single_shard"), "overview_gain_single_shard.pdf"),
        ("fanout", text("delta_fanout"), "overview_gain_fanout.pdf"),
    ):
        overview.plot_gain_by_sample(
            comparisons, metric, semantic, samples, counts, text, label, paths.figures / name
        )

    points = (
        partitions[["method", "strategy", "duplication"]]
        .merge(
            routing.loc[routing["sample"] == sample, ["method", "strategy", "fanout_mean"]],
            on=["method", "strategy"],
        )
        .merge(
            at_first_budget.loc[
                at_first_budget["sample"] == sample, ["method", "strategy", "volume", "overlap"]
            ],
            on=["method", "strategy"],
        )
    )
    overview.plot_quality_vs_cost(
        points, top_k, text, paths.figures / "overview_quality_vs_cost.pdf"
    )
    overview.plot_budget_curves(
        retrieval[retrieval["sample"] == sample],
        settings.focus_strategies,
        top_k,
        text,
        paths.figures / "overview_budget_curves.pdf",
    )
    overview.plot_slices(
        pd.read_csv(paths.metrics / "slices.csv"),
        settings.focus_strategies,
        top_k,
        text,
        paths.figures / "overview_slices.pdf",
    )
    overview.plot_sensitivity(
        pd.read_csv(paths.metrics / "sensitivity.csv"),
        settings.focus_strategies,
        top_k,
        text,
        paths.figures / "overview_sensitivity.pdf",
    )
    overview.plot_ablation_heatmaps(
        pd.read_csv(paths.metrics / "ablations.csv"),
        settings.ablation_strategies,
        top_k,
        text,
        paths.figures / "overview_ablations.pdf",
    )


def run(config: Config, paths: Paths) -> None:
    text = Labels(config.figures.language)
    partitions = pd.read_csv(paths.metrics / "partitions.csv")
    routing = pd.read_csv(paths.metrics / "routing.csv")
    retrieval = pd.read_csv(paths.metrics / "retrieval.csv", dtype={"budget": str})
    edges = pd.read_parquet(paths.graph)
    strength = node_strength(edges)

    for method in config.clustering.methods:
        _benchmark_figures(config, paths, method, text, partitions, routing, retrieval)
        _structure_figures(config, paths, method, text, edges, strength)
        logger.info("%s: figures written", method)

    sample, first_budget = config.figures.sample, str(config.evaluation.budgets[0])
    plots.plot_methods_comparison(
        retrieval[(retrieval["sample"] == sample) & (retrieval["budget"] == first_budget)],
        config.evaluation.top_k,
        text,
        text.sample(sample),
        paths.figures / "methods_overlap.pdf",
    )
    _overview_figures(config, paths, text, partitions, routing, retrieval)
    logger.info("overview figures written")


if __name__ == "__main__":
    run(*setup())
