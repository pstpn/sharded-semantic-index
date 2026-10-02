"""Locations of every artefact of the pipeline under one root directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """Artefact paths relative to ``root`` (the repository by default)."""

    root: Path = PROJECT_ROOT

    @classmethod
    def from_env(cls) -> Paths:
        """Paths under ``SSI_ROOT`` if set, else under the repository."""
        return cls(Path(os.environ.get("SSI_ROOT", PROJECT_ROOT)))

    @property
    def params(self) -> Path:
        return Path(os.environ.get("SSI_PARAMS", self.root / "params.yaml"))

    # Data
    @property
    def pairs(self) -> Path:
        return self.root / "data" / "processed" / "pairs.parquet"

    @property
    def corpus(self) -> Path:
        return self.root / "data" / "interim" / "corpus"

    @property
    def index(self) -> Path:
        return self.root / "data" / "interim" / "index"

    @property
    def queries(self) -> Path:
        return self.root / "data" / "interim" / "queries.parquet"

    @property
    def rankings(self) -> Path:
        return self.root / "data" / "interim" / "rankings"

    def assignments(self, method: str) -> Path:
        return self.root / "data" / "interim" / "assignments" / method

    def evaluation(self, method: str) -> Path:
        return self.root / "data" / "interim" / "evaluation" / method

    def verification(self, method: str) -> Path:
        return self.root / "data" / "interim" / "verification" / f"{method}.json"

    def verification_shards(self, method: str) -> Path:
        return self.root / "data" / "interim" / "verification_shards" / method

    def sensitivity(self, method: str) -> Path:
        return self.root / "data" / "interim" / "sensitivity" / f"{method}.parquet"

    def ablations(self, method: str) -> Path:
        return self.root / "data" / "interim" / "ablations" / f"{method}.parquet"

    # Models
    @property
    def graph(self) -> Path:
        return self.root / "models" / "graph.parquet"

    def clustering(self, method: str) -> Path:
        return self.root / "models" / "clusterings" / f"{method}.parquet"

    def partitions(self, method: str) -> Path:
        return self.root / "models" / "partitions" / method

    # Reports
    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"
