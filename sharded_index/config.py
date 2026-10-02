"""Typed view of ``params.yaml``, the single source of experiment settings."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from .text import Tokenizer

Algorithm = Literal["leiden", "cpm", "infomap", "metis"]

ARITY_NAMES = {2: "pairs", 3: "triples"}
REAL_SAMPLES = ("train", "holdout", "ood")
VARIABLE_SECTIONS = ("graph", "clustering", "partition", "routing")
"""Sections an ablation may override: the corpus, the index and the samples stay fixed."""

_STRATEGY_RE = re.compile(r"^(?:hash_(base|aff|bal)|(base|aff)(?:_r(\d+))?|(bal))$")


@dataclass(frozen=True)
class Strategy:
    """A parsed strategy name such as ``base``, ``aff_r3``, ``bal`` or ``hash_bal``."""

    name: str
    family: str
    replicas: int = 1
    is_hash: bool = False

    @classmethod
    def parse(cls, name: str) -> Strategy:
        """Parse a strategy name, raising ``ValueError`` for an unknown one."""
        match = _STRATEGY_RE.match(name)
        if match is None:
            msg = (
                f"unknown strategy {name!r}: expected base, aff or bal, "
                "base_r<k> or aff_r<k> with k >= 2, or hash_<family>"
            )
            raise ValueError(msg)
        hash_family, family, replicas, balanced = match.groups()
        if hash_family:
            return cls(name, hash_family, is_hash=True)
        if balanced:
            return cls(name, balanced)
        if replicas is not None and int(replicas) < 2:
            msg = f"strategy {name!r}: the replica count must be at least 2"
            raise ValueError(msg)
        return cls(name, family, replicas=int(replicas or 1))

    @property
    def hash_baseline(self) -> str:
        """Name of the random baseline with as many shards as this strategy's family."""
        return f"hash_{self.family}"


def _valid_strategies(names: tuple[str, ...]) -> tuple[str, ...]:
    for name in names:
        Strategy.parse(name)
    if len(set(names)) != len(names):
        msg = f"duplicate strategies in {names}"
        raise ValueError(msg)
    return names


StrategyNames = Annotated[tuple[str, ...], AfterValidator(_valid_strategies)]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TextConfig(_Section):
    stop_words: tuple[str, ...]


class DatasetConfig(_Section):
    name: str
    config: str | None = None
    split: str
    max_rows: int = Field(gt=0)
    docs_per_query: int = Field(gt=0)


class OodConfig(_Section):
    size: int | None = Field(gt=0)
    dataset: str
    config: str | None = None
    split: str
    column: str


class SyntheticConfig(_Section):
    size: int = Field(gt=0)
    kinds: tuple[Literal["connected", "frequent"], ...]
    arities: tuple[Literal[2, 3], ...]


class QueriesConfig(_Section):
    train_ratio: float = Field(gt=0, lt=1)
    sample_seed: int
    train_size: int | None = Field(gt=0)
    holdout_size: int | None = Field(gt=0)
    ood: OodConfig | None
    synthetic: SyntheticConfig | None

    def sample_names(self) -> list[str]:
        """Names of the evaluation samples, in report order."""
        names = ["train", "holdout"]
        if self.ood is not None:
            names.append("ood")
        if self.synthetic is not None:
            names += [
                f"{kind}_{ARITY_NAMES[arity]}"
                for kind in self.synthetic.kinds
                for arity in self.synthetic.arities
            ]
        return names


class GraphConfig(_Section):
    min_df: int = Field(ge=1)
    max_df_ratio: float = Field(gt=0, le=1)
    min_pair_count: int = Field(ge=1)
    min_npmi: float
    edge_weight: Literal["npmi", "llr", "chi2"]
    stop_words: tuple[str, ...]


class ClusteringMethod(_Section):
    algorithm: Algorithm
    resolution: float | None = Field(default=None, gt=0)
    iterations: Annotated[int, Field(ge=1)] | Literal["convergence"] | None = None
    n_parts: int | None = Field(default=None, ge=2)

    @model_validator(mode="after")
    def _parameters_match_algorithm(self) -> ClusteringMethod:
        iterative = self.algorithm in ("leiden", "cpm")
        if iterative != (self.resolution is not None) or iterative != (self.iterations is not None):
            msg = "resolution and iterations are required for leiden and cpm only"
            raise ValueError(msg)
        if (self.algorithm == "metis") != (self.n_parts is not None):
            msg = "n_parts is required for metis and not accepted otherwise"
            raise ValueError(msg)
        return self

    def with_parameter(self, name: str, value: float) -> ClusteringMethod:
        """The method with one of its own parameters changed."""
        if getattr(self, name) is None:
            msg = f"{self.algorithm} has no parameter {name!r}"
            raise ValueError(msg)
        return ClusteringMethod.model_validate({**self.model_dump(), name: value})


class ClusteringConfig(_Section):
    seed: int
    methods: dict[str, ClusteringMethod]


class BalanceConfig(_Section):
    volume_cap: float = Field(gt=0, le=1)
    max_depth: int = Field(ge=1)
    max_refinements: int = Field(ge=1)


class PartitionConfig(_Section):
    strategies: StrategyNames
    affinity_vote: Literal["llr", "raw"]
    replica_affinity_quantile: float | None = Field(ge=0, le=1)
    balance: BalanceConfig


class RoutingConfig(_Section):
    cover_weight: Literal["idf", "strength"]


class Bm25Config(_Section):
    k1: float = Field(gt=0)
    b: float = Field(ge=0, le=1)


class BootstrapConfig(_Section):
    samples: int = Field(gt=0)
    seed: int
    confidence: float = Field(gt=0, lt=1)


class ComparisonsConfig(_Section):
    pairs: tuple[tuple[str, str], ...]
    reference_method: str | None


class SlicesConfig(_Section):
    sample: str
    connectivity_quantiles: int = Field(ge=2)
    max_terms: int = Field(ge=2)


class EvaluationConfig(_Section):
    top_k: int = Field(gt=0)
    budgets: tuple[int, ...]
    ranking_depth: int = Field(gt=0)
    bootstrap: BootstrapConfig
    comparisons: ComparisonsConfig
    slices: SlicesConfig

    @model_validator(mode="after")
    def _budgets_ascend(self) -> EvaluationConfig:
        if (
            not self.budgets
            or list(self.budgets) != sorted(set(self.budgets))
            or self.budgets[0] < 1
        ):
            msg = "budgets must be a strictly ascending list of positive shard counts"
            raise ValueError(msg)
        if self.ranking_depth < self.top_k:
            msg = "ranking_depth must not be smaller than top_k"
            raise ValueError(msg)
        return self


class VerificationConfig(_Section):
    sample: str
    queries: int = Field(gt=0)
    seed: int
    score_tolerance: float = Field(gt=0)
    workers: int | None = Field(gt=0)


class SensitivityConfig(_Section):
    sample: str
    seeds: tuple[int, ...]
    train_sizes: tuple[int, ...]
    parameters: dict[str, dict[Literal["resolution", "n_parts"], tuple[float, ...]]]


class AblationsConfig(_Section):
    sample: str
    variants: dict[str, dict[str, Any]]


class DescribeConfig(_Section):
    top_clusters: int = Field(gt=0)
    top_terms: int = Field(gt=0)
    frame_word_candidates: int = Field(gt=0)


class FiguresConfig(_Section):
    sample: str
    top_clusters: int = Field(gt=0)
    graph_max_nodes: int = Field(gt=0)
    layout_seed: int


class Config(_Section):
    """All settings of the experiment."""

    text: TextConfig
    dataset: DatasetConfig
    queries: QueriesConfig
    graph: GraphConfig
    clustering: ClusteringConfig
    partition: PartitionConfig
    routing: RoutingConfig
    bm25: Bm25Config
    evaluation: EvaluationConfig
    verification: VerificationConfig
    sensitivity: SensitivityConfig
    ablations: AblationsConfig
    describe: DescribeConfig
    figures: FiguresConfig

    @model_validator(mode="after")
    def _references_resolve(self) -> Config:
        methods = set(self.clustering.methods)
        samples = set(self.queries.sample_names())
        comparisons = self.evaluation.comparisons

        _require(
            "evaluation.comparisons.pairs",
            (name for pair in comparisons.pairs for name in pair),
            set(self.partition.strategies),
        )
        if comparisons.reference_method is not None:
            _require(
                "evaluation.comparisons.reference_method", [comparisons.reference_method], methods
            )
        if set(self.sensitivity.parameters) != methods:
            msg = (
                "sensitivity.parameters must have an entry for every clustering method "
                f"({sorted(methods)}); use {{}} for a method without a parameter"
            )
            raise ValueError(msg)
        for name, sweeps in self.sensitivity.parameters.items():
            for parameter, values in sweeps.items():
                for value in values:
                    self.clustering.methods[name].with_parameter(parameter, value)
        for name, overrides in self.ablations.variants.items():
            _require(f"ablations.variants.{name}", overrides, set(VARIABLE_SECTIONS))
        for setting, sample in (
            ("evaluation.slices.sample", self.evaluation.slices.sample),
            ("verification.sample", self.verification.sample),
            ("sensitivity.sample", self.sensitivity.sample),
            ("ablations.sample", self.ablations.sample),
            ("figures.sample", self.figures.sample),
        ):
            _require(setting, [sample], samples)
        return self

    def tokenizer(self) -> Tokenizer:
        """Tokenizer of documents and queries."""
        return Tokenizer(frozenset(self.text.stop_words))

    def graph_stop_words(self) -> frozenset[str]:
        """Words kept out of the graph vocabulary: the stop list plus the graph-only list."""
        return frozenset(self.text.stop_words) | frozenset(self.graph.stop_words)

    def with_overrides(self, overrides: Mapping[str, Any]) -> Config:
        """A copy with some settings replaced, e.g. ``{"graph": {"min_npmi": 0.2}}``."""
        raw = self.model_dump()
        _merge(raw, overrides, path="")
        return Config.model_validate(raw)


def _require(setting: str, names: Iterable[str], known: set[str]) -> None:
    unknown = sorted(set(names) - known)
    if unknown:
        msg = f"{setting} refers to unknown {unknown}; known: {sorted(known)}"
        raise ValueError(msg)


def _merge(settings: dict[str, Any], overrides: Mapping[str, Any], path: str) -> None:
    """Replace the leaves of ``settings`` named by the nested ``overrides``."""
    for key, value in overrides.items():
        if key not in settings:
            msg = f"override {path}{key} does not match any setting"
            raise KeyError(msg)
        if isinstance(value, Mapping) and isinstance(settings[key], dict):
            _merge(settings[key], value, path=f"{path}{key}.")
        else:
            settings[key] = value


def load_config(path: Path) -> Config:
    """Load and validate ``params.yaml``, including every ablation variant."""
    with Path(path).open() as file:
        config = Config.model_validate(yaml.safe_load(file))
    for overrides in config.ablations.variants.values():
        config.with_overrides(overrides)
    return config
