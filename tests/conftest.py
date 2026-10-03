"""Shared fixtures: the synthetic corpus and the objects built from it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from sharded_index.config import (
    BalanceConfig,
    ClusteringMethod,
    Config,
    GraphConfig,
    PartitionConfig,
    load_config,
)
from sharded_index.data.corpus import Corpus
from sharded_index.graph.clustering import cluster_graph
from sharded_index.graph.cooccurrence import build_graph
from sharded_index.partition.model import TermPartition
from sharded_index.partition.strategies import build_strategies
from sharded_index.paths import PROJECT_ROOT
from sharded_index.text import Tokenizer
from synthetic import SEED, STOP_WORDS, STRATEGIES, make_texts, small_params


@pytest.fixture(scope="session")
def tokenizer() -> Tokenizer:
    return Tokenizer(STOP_WORDS)


@pytest.fixture(scope="session")
def texts() -> list[str]:
    return make_texts()


@pytest.fixture(scope="session")
def documents(texts: list[str]) -> dict[str, str]:
    return {f"d{i:03d}": text for i, text in enumerate(texts)}


@pytest.fixture(scope="session")
def corpus(documents: dict[str, str], tokenizer: Tokenizer) -> Corpus:
    return Corpus.from_documents(documents, tokenizer)


@pytest.fixture(scope="session")
def graph_config() -> GraphConfig:
    return GraphConfig(
        min_df=2,
        max_df_ratio=0.5,
        min_pair_count=2,
        min_npmi=0.0,
        edge_weight="npmi",
        stop_words=(),
    )


@pytest.fixture(scope="session")
def edges(texts: list[str], graph_config: GraphConfig) -> pd.DataFrame:
    return build_graph(texts, graph_config, STOP_WORDS)


@pytest.fixture(scope="session")
def leiden() -> ClusteringMethod:
    return ClusteringMethod(
        algorithm="leiden", resolution=1.0, iterations="convergence", max_iterations=100
    )


@pytest.fixture(scope="session")
def clustering(edges: pd.DataFrame, leiden: ClusteringMethod) -> dict[str, int]:
    return cluster_graph(edges, leiden, SEED)


@pytest.fixture(scope="session")
def partition_config() -> PartitionConfig:
    return PartitionConfig(
        strategies=STRATEGIES,
        affinity_vote="llr",
        replica_affinity_quantile=None,
        balance=BalanceConfig(volume_cap=0.3, max_depth=12, max_refinements=5),
    )


@pytest.fixture(scope="session")
def strategies(
    clustering: dict[str, int],
    edges: pd.DataFrame,
    corpus: Corpus,
    leiden: ClusteringMethod,
    partition_config: PartitionConfig,
) -> dict[str, TermPartition]:
    return build_strategies(clustering, edges, corpus, leiden, SEED, partition_config)


@pytest.fixture(scope="session")
def project_config() -> Config:
    """The configuration of the experiment as committed in ``params.yaml``."""
    return load_config(PROJECT_ROOT / "params.yaml")


@pytest.fixture
def small_config(tmp_path: Path) -> Config:
    """The project configuration scaled down to the synthetic data."""
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(small_params(), sort_keys=False))
    return load_config(path)
