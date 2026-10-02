"""Plumbing shared by the stage entry points: setup, arguments and artefact access."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from sharded_index.config import Config, load_config
from sharded_index.data.msmarco import docs_from_pairs
from sharded_index.data.queries import queries_of
from sharded_index.index.ranking import Rankings
from sharded_index.partition.model import TermPartition
from sharded_index.paths import Paths


def setup() -> tuple[Config, Paths]:
    """Configure logging and load the settings of the run."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    paths = Paths.from_env()
    return load_config(paths.params), paths


def method_argument(config: Config) -> str:
    """The ``--method`` command-line argument of a per-method stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(config.clustering.methods))
    return str(parser.parse_args().method)


def read_documents(paths: Paths) -> dict[str, str]:
    """``doc_id → text`` of the corpus."""
    return docs_from_pairs(pd.read_parquet(paths.pairs, columns=["doc_id", "doc_text"]))


def read_clustering(paths: Paths, method: str) -> dict[str, int]:
    table = pd.read_parquet(paths.clustering(method))
    return dict(zip(table["term"].tolist(), table["cluster"].tolist(), strict=True))


def read_partitions(config: Config, paths: Paths, method: str) -> dict[str, TermPartition]:
    """Every configured strategy of a method."""
    directory = paths.partitions(method)
    return {
        name: TermPartition.load(directory / f"{name}.parquet")
        for name in config.partition.strategies
    }


def read_samples(config: Config, paths: Paths) -> dict[str, list[str]]:
    """Sample name → its queries, in report order."""
    table = pd.read_parquet(paths.queries)
    return {sample: queries_of(table, sample) for sample in config.queries.sample_names()}


def read_rankings(config: Config, paths: Paths) -> dict[str, Rankings]:
    """Reference rankings of every sample."""
    return {
        sample: Rankings.load(paths.rankings / f"{sample}.npz")
        for sample in config.queries.sample_names()
    }
