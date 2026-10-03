"""Synthetic data of the test suite: four topics of five words each."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yaml

from sharded_index.paths import PROJECT_ROOT

TOPICS = [
    ["apple", "banana", "fruit", "juice", "sweet"],
    ["dog", "cat", "pet", "animal", "fur"],
    ["python", "code", "bug", "program", "compile"],
    ["ocean", "wave", "beach", "sand", "swim"],
]
STOP_WORDS = frozenset({"the", "what", "is"})
SEED = 42
STRATEGIES = (
    "hash_base",
    "base",
    "base_r2",
    "base_r3",
    "hash_aff",
    "aff",
    "aff_r2",
    "aff_r3",
    "hash_bal",
    "bal",
)
QUERIES = [
    "apple juice sweet",
    "dog cat pet",
    "python bug windows10",
    "ocean swim",
    "zzz nothing matches",
]


def make_texts(n: int = 400) -> list[str]:
    """Texts of two to four words of one topic.

    Some also get a stop word, a digit token (never a graph term) or a word
    of the next topic (so that clusters have edges between them).
    """
    rng = np.random.default_rng(0)
    texts = []
    for i in range(n):
        words = list(rng.choice(TOPICS[i % 4], size=rng.integers(2, 5), replace=False))
        if i % 7 == 0:
            words.append("the")
        if i % 5 == 0:
            words.append("windows10")
        if i % 6 == 0:
            words.append(str(rng.choice(TOPICS[(i + 1) % 4])))
        rng.shuffle(words)
        texts.append(" ".join(words))
    return texts


def make_pairs(n_queries: int = 300) -> pd.DataFrame:
    """Query-passage pairs in the format of the extraction stage: two passages per query."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n_queries):
        topic = TOPICS[i % 4]
        query = " ".join(rng.choice(topic, size=rng.integers(2, 4), replace=False))
        for j in range(2):
            words = [*rng.choice(topic, size=4, replace=False), f"item{i % 23}", f"note{i}x{j}"]
            if (i + j) % 6 == 0:
                words.append(str(rng.choice(TOPICS[(i + 1) % 4])))
            rows.append((query, f"{i:04d}{j}", " ".join(words), j == 0))
    return pd.DataFrame(rows, columns=["query", "doc_id", "doc_text", "is_selected"])


def small_params() -> dict[str, Any]:
    """``params.yaml`` scaled down to the synthetic data."""
    params: dict[str, Any] = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text())
    params["text"]["stop_words"] = sorted(STOP_WORDS)
    params["queries"].update(train_ratio=0.8, train_size=40, ood=None)
    params["queries"]["synthetic"]["size"] = 10
    params["graph"]["stop_words"] = []
    params["clustering"]["methods"]["metis"]["n_parts"] = 4
    params["partition"]["balance"]["volume_cap"] = 0.3
    params["evaluation"]["bootstrap"]["samples"] = 20
    params["evaluation"]["ranking_depth"] = 50
    params["verification"].update(queries=2, workers=1)
    params["sensitivity"].update(
        seeds=[1],
        train_sizes=[60],
        parameters={
            "leiden": {"resolution": [2]},
            "cpm": {},
            "infomap": {},
            "metis": {"n_parts": [6]},
        },
    )
    params["ablations"]["variants"] = {
        "baseline": {},
        "vote_raw": {"partition": {"affinity_vote": "raw"}},
        "cpm_fine": {
            "clustering": {
                "methods": {"cpm": {"resolution": 0.1, "iterations": 2, "max_iterations": None}}
            }
        },
    }
    params["describe"].update(top_clusters=3, top_terms=3, frame_word_candidates=5)
    params["figures"].update(top_clusters=3, graph_max_nodes=50)
    return params
