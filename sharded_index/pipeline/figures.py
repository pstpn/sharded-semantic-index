"""Draw the figures: per-method cluster-structure plots, then the overview figures."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sharded_index.config import Config
from sharded_index.graph.cooccurrence import node_strength
from sharded_index.paths import Paths
from sharded_index.pipeline.common import read_clustering, setup
from sharded_index.reporting import overview, plots
from sharded_index.reporting.labels import Labels

logger = logging.getLogger(__name__)


def _method_figures(
    config: Config,
    paths: Paths,
    method: str,
    text: Labels,
    root: Path,
    partitions: pd.DataFrame,
    routing: pd.DataFrame,
    retrieval: pd.DataFrame,
    edges: pd.DataFrame,
    strength: dict[str, float],
) -> None:
    """The storage/routing trade-off of every strategy and the cluster structure of one method."""
    settings = config.figures
    sample, budgets = settings.sample, settings.budgets
    strategies = list(config.partition.strategies)
    target = root / method

    retrieval = retrieval[(retrieval["method"] == method) & (retrieval["sample"] == sample)]
    overlap = retrieval.pivot(index="strategy", columns="budget", values="overlap")
    fanout = routing[(routing["method"] == method) & (routing["sample"] == sample)].set_index(
        "strategy"
    )["fanout_mean"]
    duplication = partitions[partitions["method"] == method].set_index("strategy")["duplication"]
    points = pd.DataFrame(
        {
            "strategy": strategies,
            "duplication": duplication.loc[strategies].to_numpy(),
            "fanout_mean": fanout.loc[strategies].to_numpy(),
            "label": [
                name
                + "\n"
                + ", ".join(f"@{budget} {overlap.loc[name, str(budget)]:.2f}" for budget in budgets)
                for name in strategies
            ],
        }
    )
    plots.plot_duplication_vs_fanout(points, text, target / "duplication_vs_fanout.pdf")

    top, seed = settings.top_clusters, settings.layout_seed
    clustering = read_clustering(paths, method)
    plots.plot_cluster_sizes(clustering, top, text, target / "cluster_sizes.pdf")
    plots.plot_cluster_wordclouds(
        clustering, strength, top, seed, text, target / "cluster_wordclouds.pdf"
    )
    plots.plot_graph_clusters(
        edges, clustering, settings.graph_max_nodes, seed, target / "graph_clusters.pdf"
    )
    plots.plot_cluster_heatmap(edges, clustering, top, text, target / "cluster_heatmap.pdf")


def _overview_figures(
    config: Config,
    paths: Paths,
    text: Labels,
    root: Path,
    partitions: pd.DataFrame,
    routing: pd.DataFrame,
    retrieval: pd.DataFrame,
    fanouts: dict[str, dict[str, pd.Series]],
) -> None:
    """Every method together: the unbudgeted regime, then the focus strategies against hash."""
    settings = config.figures
    sample, top_k, budgets = settings.sample, config.evaluation.top_k, list(settings.budgets)
    focus = list(settings.focus_strategies)
    samples = [name for name in config.queries.sample_names() if name in set(routing["sample"])]
    counts = routing.drop_duplicates("sample").set_index("sample")["evaluated"].to_dict()
    sensitivity = pd.read_csv(paths.metrics / "sensitivity.csv")
    ablations = pd.read_csv(paths.metrics / "ablations.csv")

    costs = partitions[["method", "strategy", "duplication"]].merge(
        routing.loc[routing["sample"] == sample, ["method", "strategy", "fanout_mean"]],
        on=["method", "strategy"],
    )
    points = {
        budget: costs.merge(
            retrieval.loc[
                (retrieval["sample"] == sample) & (retrieval["budget"] == str(budget)),
                ["method", "strategy", "volume", "overlap"],
            ],
            on=["method", "strategy"],
        )
        for budget in budgets
    }
    overview.plot_quality_vs_cost(points, top_k, text, root / "overview_quality_vs_cost.pdf")
    overview.plot_parameter_sweeps(
        sensitivity,
        [settings.unbudgeted_strategy, *focus],
        config.partition.balance.volume_cap,
        budgets[0],
        top_k,
        text,
        root / "overview_parameter_sweeps.pdf",
    )
    overview.plot_variant_heatmap(
        ablations,
        settings.unbudgeted_strategy,
        "largest_shard",
        "value",
        text("largest_shard_pct"),
        root / "overview_variants_largest_shard.pdf",
    )

    overview.plot_budget_curves(
        retrieval[retrieval["sample"] == sample],
        focus,
        top_k,
        text,
        root / "overview_budget_curves.pdf",
    )
    overview.plot_by_sample(routing, focus, samples, counts, text, root / "overview_by_sample.pdf")
    overview.plot_fanout_ecdf(fanouts, text, root / "overview_fanout_ecdf.pdf")
    overview.plot_slices(
        pd.read_csv(paths.metrics / "slices.csv"),
        focus,
        top_k,
        budgets,
        text,
        root / "overview_slices.pdf",
    )
    overview.plot_variant_heatmap(
        ablations,
        focus[0],
        f"overlap_{budgets[0]}",
        "delta",
        text("delta_pp", k=top_k, shards=text.shards(budgets[0])),
        root / "overview_variants_overlap.pdf",
    )


def _fanouts(config: Config, paths: Paths) -> dict[str, dict[str, pd.Series]]:
    """Per-query cover sizes on the figures' sample: method -> strategy -> values."""
    strategies = overview.with_baselines(config.figures.focus_strategies)
    fanouts: dict[str, dict[str, pd.Series]] = {}
    for method in config.clustering.methods:
        queries = pd.read_parquet(paths.evaluation(method) / "queries.parquet")
        measured = queries[(queries["sample"] == config.figures.sample) & queries["evaluated"]]
        fanouts[method] = {
            name: measured.loc[measured["strategy"] == name, "fanout"] for name in strategies
        }
    return fanouts


def run(config: Config, paths: Paths) -> None:
    partitions = pd.read_csv(paths.metrics / "partitions.csv")
    routing = pd.read_csv(paths.metrics / "routing.csv")
    retrieval = pd.read_csv(paths.metrics / "retrieval.csv", dtype={"budget": str})
    edges = pd.read_parquet(paths.graph)
    strength = node_strength(edges)
    fanouts = _fanouts(config, paths)

    for language in config.figures.languages:
        text = Labels(language)
        root = paths.figures / language
        for method in config.clustering.methods:
            _method_figures(
                config, paths, method, text, root, partitions, routing, retrieval, edges, strength
            )
        _overview_figures(config, paths, text, root, partitions, routing, retrieval, fanouts)
        logger.info("%s: figures written", language)


if __name__ == "__main__":
    run(*setup())
