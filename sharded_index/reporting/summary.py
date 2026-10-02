"""The main results table: every method and strategy on one query sample.

Run:  python -m sharded_index.reporting.summary [--sample holdout]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sharded_index.paths import Paths


def summary_table(metrics: Path, sample: str) -> pd.DataFrame:
    """Partition cost, cover size and overlap by budget, one row per ``(method, strategy)``."""
    partitions = pd.read_csv(metrics / "partitions.csv")
    routing = pd.read_csv(metrics / "routing.csv")
    retrieval = pd.read_csv(metrics / "retrieval.csv", dtype={"budget": str})

    overlap = (
        retrieval[retrieval["sample"] == sample]
        .pivot(index=["method", "strategy"], columns="budget", values="overlap")
        .add_prefix("overlap_")
        .reset_index()
    )
    columns = ["method", "strategy", "fanout_mean", "single_shard_share"]
    return (
        partitions[["method", "strategy", "shards", "duplication", "largest_shard"]]
        .merge(routing.loc[routing["sample"] == sample, columns], on=["method", "strategy"])
        .merge(overlap, on=["method", "strategy"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", default="holdout")
    sample = parser.parse_args().sample
    table = summary_table(Paths.from_env().metrics, sample)
    sys.stdout.write(table.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
