"""Draw the figures: per-method benchmark and cluster-structure plots and a method comparison."""

from __future__ import annotations

import logging

import pandas as pd

from sharded_index.config import REAL_SAMPLES, Config
from sharded_index.graph.cooccurrence import node_strength
from sharded_index.paths import Paths
from sharded_index.pipeline.common import read_clustering, setup
from sharded_index.reporting import plots

logger = logging.getLogger(__name__)


def _benchmark_figures(
    config: Config,
    paths: Paths,
    method: str,
    partitions: pd.DataFrame,
    routing: pd.DataFrame,
    retrieval: pd.DataFrame,
) -> None:
    """Cover size, overlap by budget and the storage/routing trade-off of one method."""
    sample = config.figures.sample
    strategies = list(config.partition.strategies)
    target = paths.figures / method

    queries = pd.read_parquet(paths.evaluation(method) / "queries.parquet")
    measured = queries[(queries["sample"] == sample) & queries["evaluated"]]
    fanouts = {
        name: measured.loc[measured["strategy"] == name, "fanout"].to_numpy() for name in strategies
    }
    plots.plot_fanout_ecdf(
        fanouts, f"Query cover size: {method}, {sample}", target / "fanout_ecdf.pdf"
    )

    retrieval = retrieval[(retrieval["method"] == method) & (retrieval["sample"] == sample)]
    overlap = retrieval.pivot(index="strategy", columns="budget", values="overlap")
    budgets = list(dict.fromkeys(retrieval["budget"]))
    plots.plot_overlap_by_budget(
        {name: overlap.loc[name, budgets].to_dict() for name in strategies},
        config.evaluation.top_k,
        f"Overlap by shard budget: {method}, {sample}",
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
    plots.plot_duplication_vs_fanout(
        points,
        f"Duplication against cover size: {method}, {sample}",
        target / "duplication_vs_fanout.pdf",
    )

    real_samples = [name for name in config.queries.sample_names() if name in REAL_SAMPLES]
    plots.plot_fanout_by_sample(
        fanout.loc[strategies, real_samples],
        f"Query cover size by sample: {method}",
        target / "fanout_by_sample.pdf",
    )


def _structure_figures(
    config: Config,
    paths: Paths,
    method: str,
    edges: pd.DataFrame,
    strength: dict[str, float],
) -> None:
    """Sizes, contents and mutual links of the clusters of one method."""
    settings = config.figures
    top, seed = settings.top_clusters, settings.layout_seed
    target = paths.figures / method
    clustering = read_clustering(paths, method)

    plots.plot_cluster_sizes(
        clustering, top, f"Cluster sizes: {method}", target / "cluster_sizes.pdf"
    )
    plots.plot_cluster_wordclouds(
        clustering,
        strength,
        top,
        seed,
        f"Largest clusters: {method}",
        target / "cluster_wordclouds.pdf",
    )
    plots.plot_graph_clusters(
        edges,
        clustering,
        settings.graph_max_nodes,
        seed,
        f"Term graph coloured by cluster: {method}",
        target / "graph_clusters.pdf",
    )
    plots.plot_tsne(
        edges,
        clustering,
        strength,
        top,
        seed,
        f"t-SNE of graph terms: {method}",
        target / "tsne.pdf",
    )
    plots.plot_cluster_heatmap(
        edges,
        clustering,
        top,
        f"Edge weight between clusters: {method}",
        target / "cluster_heatmap.pdf",
    )


def run(config: Config, paths: Paths) -> None:
    partitions = pd.read_csv(paths.metrics / "partitions.csv")
    routing = pd.read_csv(paths.metrics / "routing.csv")
    retrieval = pd.read_csv(paths.metrics / "retrieval.csv", dtype={"budget": str})
    edges = pd.read_parquet(paths.graph)
    strength = node_strength(edges)

    for method in config.clustering.methods:
        _benchmark_figures(config, paths, method, partitions, routing, retrieval)
        _structure_figures(config, paths, method, edges, strength)
        logger.info("%s: figures written", method)

    sample, first_budget = config.figures.sample, str(config.evaluation.budgets[0])
    plots.plot_methods_comparison(
        retrieval[(retrieval["sample"] == sample) & (retrieval["budget"] == first_budget)],
        f"Overlap@{config.evaluation.top_k} with {first_budget} shard probed: {sample}",
        paths.figures / "methods_overlap.pdf",
    )


if __name__ == "__main__":
    run(*setup())
