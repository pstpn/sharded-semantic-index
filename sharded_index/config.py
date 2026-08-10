"""Project configuration: artefact paths, partition parameters, params.yaml access.

Paths follow the cookiecutter-data-science layout and are anchored to the
project root, so imports work from any working directory (notebooks included).
``SSI_ROOT`` relocates every artefact (the smoke test runs the pipeline in a
temporary directory this way); ``SSI_PARAMS`` points to an alternative
``params.yaml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJ_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("SSI_ROOT", PROJ_ROOT))

# ── Data artefacts (gitignored, produced by the DVC pipeline) ─────────
DATA_DIR = ROOT / "data"
PAIRS_PATH = DATA_DIR / "processed" / "pairs.parquet"
INTERIM_DIR = DATA_DIR / "interim"
TERM_DF_PATH = INTERIM_DIR / "term_df.parquet"    # документная частота термов
ASSIGNMENTS_DIR = INTERIM_DIR / "assignments"     # {strategy}.parquet: (doc_id, shard)
INDICES_DIR = INTERIM_DIR / "indices"             # hash/, leiden/, leiden_r{k}/, full/
EVAL_DIR = INTERIM_DIR / "eval"                   # fanouts.parquet

# ── Models: обученные разбиения термов ────────────────────────────────
MODELS_DIR = ROOT / "models"
PARTITIONS_DIR = MODELS_DIR / "partitions"        # leiden_core/, hash/, leiden/, leiden_r{k}/

# ── Reporting ─────────────────────────────────────────────────────────
METRICS_DIR = ROOT / "metrics"                    # metrics.json (git, dvc metrics)
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"             # фигуры (PDF, git)


@dataclass
class PartitionConfig:
    """Parameters of the NPMI graph and its Leiden clustering."""

    # ── NPMI co-occurrence graph ──────────────────────────────────────
    min_df: int = 2
    """Minimum document frequency for a term to enter the graph."""

    max_df_ratio: float = 0.5
    """Terms present in more than this fraction of documents are dropped."""

    min_pair_count: int = 2
    """Minimum co-occurrence count for an edge to be kept."""

    min_npmi: float = 0.0
    """Minimum NPMI weight for an edge to be kept."""

    edge_weight: str = "npmi"
    """Edge weighting scheme: ``npmi`` | ``llr`` | ``chi2`` (topology stays NPMI-filtered)."""

    graph_stop_words: tuple[str, ...] = ()
    """Extra stop list for the graph vocabulary only (question-frame words etc.)."""

    # ── Clustering ────────────────────────────────────────────────────
    cluster_method: str = "leiden"
    """``leiden`` | ``cpm`` | ``infomap`` | ``metis`` (needs pymetis)."""

    leiden_resolution: float = 1.0
    """Leiden resolution parameter. Higher → more, smaller clusters."""

    leiden_seed: int = 42
    """Random seed for reproducible clustering."""


def load_params() -> dict:
    """Load ``params.yaml`` (or the file named by ``SSI_PARAMS``)."""
    params_path = Path(os.environ.get("SSI_PARAMS", ROOT / "params.yaml"))
    with open(params_path) as f:
        return yaml.safe_load(f)


_NON_GRAPH_KNOBS = (
    "term_replicas",
    "replica_affinity_quantile",
    "doc_affinity_fallback",
    "volume_cap_ratio",
)
"""Ключи секции ``partition``, не являющиеся параметрами графа/кластеризации."""


def partition_config(params: dict) -> PartitionConfig:
    """Build :class:`PartitionConfig` from the ``partition`` params section."""
    fields = {
        k: v for k, v in params["partition"].items()
        if k not in _NON_GRAPH_KNOBS
    }
    if fields.get("graph_stop_words") is None:
        fields.pop("graph_stop_words")
    elif "graph_stop_words" in fields:
        fields["graph_stop_words"] = tuple(fields["graph_stop_words"])
    return PartitionConfig(**fields)


def strategy_names(params: dict) -> list[str]:
    """Benchmarked strategies (= artefact subdirectory names).

    Always ``hash`` and ``leiden`` (single assignment), plus ``leiden_r{k}``
    for every replica count in ``partition.term_replicas``.  With
    ``partition.doc_affinity_fallback`` on, the same family is repeated with
    the document-affinity fallback: ``leiden_aff``, ``leiden_aff_r{k}``.
    With ``partition.volume_cap_ratio`` set (needs the affinity fallback),
    the volume-balanced family is added: ``leiden_bal`` (capped shards) and
    ``leiden_bal_r{k}`` (volume-guarded replication).
    """
    partition = params["partition"]
    replicas = partition.get("term_replicas", [])
    names = ["hash", "leiden"] + [f"leiden_r{k}" for k in replicas]
    if partition.get("doc_affinity_fallback", False):
        names += ["leiden_aff"] + [f"leiden_aff_r{k}" for k in replicas]
        if partition.get("volume_cap_ratio"):
            names += ["leiden_bal"] + [f"leiden_bal_r{k}" for k in replicas]
    return names
