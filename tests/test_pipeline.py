"""The whole pipeline on synthetic data: every stage runs and the outputs are consistent."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from sharded_index.config import Config
from sharded_index.paths import PROJECT_ROOT, Paths
from sharded_index.pipeline import (
    ablations,
    assign_documents,
    build_corpus,
    build_graph,
    build_index,
    build_partitions,
    cluster,
    describe,
    evaluate,
    figures,
    rank_queries,
    report,
    sample_queries,
    sensitivity,
    verify,
)
from synthetic import make_pairs

SHARED_STAGES = (build_corpus, build_index, build_graph, sample_queries, rank_queries)
METHOD_STAGES = (cluster, build_partitions, assign_documents, evaluate, sensitivity, ablations)
OUTPUTS = (PROJECT_ROOT / "docs" / "outputs.md").read_text()


def _documented_section(file_name: str) -> str:
    """The part of ``docs/outputs.md`` that describes one file of ``metrics/``."""
    start = OUTPUTS.index(f"### `metrics/{file_name}`")
    end = OUTPUTS.find("\n##", start + 1)
    return OUTPUTS[start:end]


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    paths = Paths(tmp_path)
    paths.pairs.parent.mkdir(parents=True)
    make_pairs().to_parquet(paths.pairs, index=False)
    return paths


def test_pipeline_runs_end_to_end(small_config: Config, paths: Paths) -> None:
    config = small_config
    methods = list(config.clustering.methods)
    strategies = list(config.partition.strategies)
    samples = config.queries.sample_names()
    budgets = len(config.evaluation.budgets) + 1

    for stage in SHARED_STAGES:
        stage.run(config, paths)
    for method in methods:
        for stage in METHOD_STAGES:
            stage.run(config, paths, method)
        assert verify.run(config, paths, method)
    report.run(config, paths)
    describe.run(config, paths)
    figures.run(config, paths)

    def table(name: str) -> pd.DataFrame:
        return pd.read_csv(paths.metrics / f"{name}.csv", dtype={"budget": str})

    configurations = len(methods) * len(strategies)
    assert len(table("partitions")) == configurations
    assert len(table("routing")) == configurations * len(samples)

    retrieval = table("retrieval")
    assert len(retrieval) == configurations * len(samples) * budgets
    full_cover = retrieval[retrieval["budget"] == "full"]
    assert (full_cover["overlap"] == 1.0).all()
    assert (retrieval["overlap_ci_low"] <= retrieval["overlap"]).all()
    assert (retrieval["overlap"] <= retrieval["overlap_ci_high"]).all()
    by_budget = retrieval.pivot(
        index=["method", "strategy", "sample"], columns="budget", values="overlap"
    )
    assert (by_budget["1"] <= by_budget["2"]).all()

    comparisons = table("comparisons")
    assert set(comparisons["comparison"]) == {"hash", "strategy", "method"}
    assert set(comparisons.loc[comparisons["comparison"] == "method", "baseline_method"]) == {
        config.evaluation.comparisons.reference_method
    }

    verification = table("verification")
    assert len(verification) == configurations
    assert verification["passed"].all()
    assert verification["assignment_exact"].all()
    assert (verification["scores_compared"] > 0).all()

    sensitivity_table = table("sensitivity")
    assert set(sensitivity_table["sweep"]) == {"seed", "train_size", "resolution", "n_parts"}
    assert set(sensitivity_table.loc[sensitivity_table["sweep"] == "n_parts", "method"]) == {
        "metis"
    }
    ablations_table = table("ablations")
    assert set(ablations_table["variant"]) == set(config.ablations.variants)
    shared = ablations_table[ablations_table["variant"] != "cpm_fine"]
    assert len(shared) == 2 * configurations
    assert set(ablations_table.loc[ablations_table["variant"] == "cpm_fine", "method"]) == {"cpm"}

    main = table("routing").query("sample == @config.sensitivity.sample")
    baseline = ablations_table[ablations_table["variant"] == "baseline"]
    merged = baseline.merge(main, on=["method", "strategy"], suffixes=("", "_main"))
    assert (merged["fanout_mean"] == merged["fanout_mean_main"]).all()

    slices = table("slices")
    assert set(slices["slicing"]) == {"connectivity", "terms"}
    by_length = slices[slices["slicing"] == "terms"]
    assert (by_length["fanout_mean"] <= by_length["known_terms_mean"]).all()
    evaluated = table("routing").query("sample == @config.evaluation.slices.sample")
    per_slicing = slices.groupby(["method", "strategy", "slicing"])["queries"].sum()
    assert set(per_slicing) == set(evaluated["evaluated"])
    assert len(table("clusterings")) == len(methods)
    corpus = json.loads((paths.metrics / "corpus.json").read_text())
    assert corpus["collection"]["documents"] == make_pairs()["doc_id"].nunique()

    for path in sorted(paths.metrics.glob("*.csv")):
        described = _documented_section(path.name)
        for column in pd.read_csv(path).columns:
            name = re.sub(r"^(overlap|volume)_(\d+|full)$", r"\1_<b>", column)
            assert f"`{name}`" in described, f"{path.name}: {column} is not described"
    for section, fields in corpus.items():
        for field in fields:
            assert f"`{field}`" in _documented_section("corpus.json"), f"corpus.json: {section}"

    overview_names = {path.name for path in paths.figures.glob("*.pdf")}
    assert overview_names == {
        "methods_overlap.pdf",
        "overview_single_shard_by_sample.pdf",
        "overview_overlap_by_sample.pdf",
        "overview_fanout_by_sample.pdf",
        "overview_gain_overlap.pdf",
        "overview_gain_single_shard.pdf",
        "overview_gain_fanout.pdf",
        "overview_quality_vs_cost.pdf",
        "overview_budget_curves.pdf",
        "overview_slices.pdf",
        "overview_sensitivity.pdf",
        "overview_ablations.pdf",
    }
    assert all(f"`reports/figures/{name}`" in OUTPUTS for name in overview_names)
    for method in methods:
        figure_names = {path.name for path in (paths.figures / method).glob("*.pdf")}
        assert len(figure_names) == 9
        assert all(f"`reports/figures/<method>/{name}`" in OUTPUTS for name in figure_names)


def test_artefacts_are_byte_stable(small_config: Config, paths: Paths, tmp_path: Path) -> None:
    second = Paths(tmp_path / "second")
    second.pairs.parent.mkdir(parents=True)
    make_pairs().to_parquet(second.pairs, index=False)

    for run in (paths, second):
        for stage in (build_corpus, build_graph, sample_queries):
            stage.run(small_config, run)
        for stage in (cluster, build_partitions, assign_documents):
            stage.run(small_config, run, "leiden")

    for artefact in (
        paths.corpus / "doc_terms.npz",
        paths.graph,
        paths.queries,
        paths.clustering("leiden"),
        paths.partitions("leiden") / "bal.parquet",
        paths.assignments("leiden") / "bal.npz",
    ):
        relative = artefact.relative_to(paths.root)
        assert artefact.read_bytes() == (second.root / relative).read_bytes(), relative


def test_verification_failure_is_reported(small_config: Config, paths: Paths) -> None:
    for stage in SHARED_STAGES:
        stage.run(small_config, paths)
    for stage in (cluster, build_partitions, assign_documents):
        stage.run(small_config, paths, "leiden")

    target = paths.assignments("leiden") / "bal.npz"
    target.write_bytes((paths.assignments("leiden") / "hash_bal.npz").read_bytes())
    assert not verify.run(small_config, paths, "leiden")
    result = json.loads(paths.verification("leiden").read_text())
    assert not result["strategies"]["bal"]["completeness"]["assignment_exact"]
    assert result["strategies"]["aff"]["passed"]
