from __future__ import annotations

import pytest
from pydantic import ValidationError

from sharded_index.config import ClusteringMethod, Config, Strategy


def test_project_params_are_valid(project_config: Config) -> None:
    assert set(project_config.clustering.methods) == {"leiden", "cpm", "infomap", "metis"}
    assert project_config.queries.sample_names() == [
        "train",
        "holdout",
        "ood",
        "connected_pairs",
        "connected_triples",
        "frequent_pairs",
        "frequent_triples",
    ]


def test_yaml_booleans_are_not_stop_words(project_config: Config) -> None:
    assert {"on", "off", "no", "y"} <= set(project_config.text.stop_words)


@pytest.mark.parametrize(
    ("name", "family", "replicas", "is_hash"),
    [
        ("base", "base", 1, False),
        ("aff_r3", "aff", 3, False),
        ("bal", "bal", 1, False),
        ("hash_bal", "bal", 1, True),
    ],
)
def test_strategy_names_parse(name: str, family: str, replicas: int, is_hash: bool) -> None:
    strategy = Strategy.parse(name)
    assert (strategy.family, strategy.replicas, strategy.is_hash) == (family, replicas, is_hash)
    assert strategy.hash_baseline == f"hash_{family}"


@pytest.mark.parametrize("name", ["bal_r2", "leiden", "base_r1", "hash", "hash_aff_r2"])
def test_unknown_strategy_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="strategy"):
        Strategy.parse(name)


def test_method_parameters_must_match_the_algorithm() -> None:
    with pytest.raises(ValidationError):
        ClusteringMethod(algorithm="leiden")
    with pytest.raises(ValidationError):
        ClusteringMethod(algorithm="leiden", resolution=1.0)
    with pytest.raises(ValidationError):
        ClusteringMethod(algorithm="cpm", resolution=1.0, iterations=0)
    with pytest.raises(ValidationError):
        ClusteringMethod(algorithm="metis", n_parts=8, iterations=2)
    assert ClusteringMethod(algorithm="cpm", resolution=1.0, iterations=3).iterations == 3
    with pytest.raises(ValidationError, match="max_iterations"):
        ClusteringMethod(algorithm="leiden", resolution=1.0, iterations="convergence")
    with pytest.raises(ValidationError, match="max_iterations"):
        ClusteringMethod(algorithm="cpm", resolution=1.0, iterations=3, max_iterations=10)
    converging = ClusteringMethod(
        algorithm="leiden", resolution=1.0, iterations="convergence", max_iterations=10
    )
    assert converging.max_iterations == 10
    with pytest.raises(ValidationError):
        ClusteringMethod(algorithm="infomap", resolution=1.0)
    with pytest.raises(ValidationError):
        ClusteringMethod(algorithm="metis", resolution=1.0)
    with pytest.raises(ValueError, match="no parameter"):
        ClusteringMethod(algorithm="infomap").with_parameter("resolution", 2.0)
    assert (
        ClusteringMethod(algorithm="metis", n_parts=8).with_parameter("n_parts", 16.0).n_parts == 16
    )


def test_overrides_replace_nested_settings(project_config: Config) -> None:
    varied = project_config.with_overrides({"graph": {"min_npmi": 0.2, "stop_words": []}})
    assert varied.graph.min_npmi == 0.2
    assert varied.graph.stop_words == ()
    assert varied.graph.min_df == project_config.graph.min_df
    assert project_config.graph.min_npmi == 0.0


def test_unknown_settings_are_rejected(project_config: Config) -> None:
    with pytest.raises(KeyError, match="min_npmy"):
        project_config.with_overrides({"graph": {"min_npmy": 0.2}})
    raw = project_config.model_dump()
    raw["graph"]["typo"] = 1
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_references_between_sections_are_checked(project_config: Config) -> None:
    raw = project_config.model_dump()
    raw["figures"]["sample"] = "missing"
    with pytest.raises(ValidationError, match=r"figures\.sample"):
        Config.model_validate(raw)

    raw = project_config.model_dump()
    raw["sensitivity"]["parameters"]["infomap"] = {"resolution": [2.0]}
    with pytest.raises(ValidationError, match="no parameter"):
        Config.model_validate(raw)

    raw = project_config.model_dump()
    del raw["sensitivity"]["parameters"]["infomap"]
    with pytest.raises(ValidationError, match="every clustering method"):
        Config.model_validate(raw)

    raw = project_config.model_dump()
    raw["ablations"]["variants"]["corpus"] = {"text": {"stop_words": []}}
    with pytest.raises(ValidationError, match=r"ablations\.variants\.corpus"):
        Config.model_validate(raw)
