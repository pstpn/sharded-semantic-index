from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sharded_index.config import ComparisonsConfig
from sharded_index.data.corpus import Corpus
from sharded_index.graph.cooccurrence import node_strength
from sharded_index.reporting import plots
from sharded_index.reporting.describe import (
    clusters_table,
    collection_description,
    frame_word_candidates,
    graph_description,
)
from sharded_index.reporting.tables import comparison_pairs, variants_table, write_table
from sharded_index.text import Tokenizer
from synthetic import STRATEGIES


def test_tables_are_written_with_fixed_precision(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "strategy": ["bal", "hash_bal"],
            "shards": [133.0, 140.0],
            "hash_terms": [0.0, np.nan],
            "duplication": [19.81234567, 23.5],
            "max_score_difference": [3.5527e-15, 0.0],
        }
    )
    write_table(frame, tmp_path / "nested" / "table.csv")
    assert (tmp_path / "nested" / "table.csv").read_text().splitlines() == [
        "strategy,shards,hash_terms,duplication,max_score_difference",
        "bal,133,0,19.8123,3.5527e-15",
        "hash_bal,140,,23.5,0.0",
    ]


def test_variant_tables_are_stacked_with_readable_values() -> None:
    parts = {
        "leiden": pd.DataFrame(
            {"sweep": ["seed", "resolution"], "value": [42.0, 0.5], "x": [1, 2]}
        ),
        "cpm": pd.DataFrame({"sweep": ["seed"], "value": [1.0], "x": [3]}),
    }
    table = variants_table(parts, ["sweep", "method", "value"])
    assert table.columns.tolist() == ["sweep", "method", "value", "x"]
    assert table["value"].tolist() == ["42", "0.5", "1"]
    assert table["method"].tolist() == ["leiden", "leiden", "cpm"]


def test_comparison_pairs_cover_baselines_strategies_and_methods() -> None:
    comparisons = ComparisonsConfig(pairs=(("bal", "aff"),), reference_method="leiden")
    pairs = comparison_pairs(["leiden", "cpm"], STRATEGIES, comparisons)
    kinds = [kind for kind, _, _ in pairs]
    assert kinds.count("hash") == 2 * 7
    assert kinds.count("strategy") == 2
    assert kinds.count("method") == len(STRATEGIES)
    assert ("hash", ("cpm", "aff_r3"), ("cpm", "hash_aff")) in pairs
    assert ("strategy", ("leiden", "bal"), ("leiden", "aff")) in pairs
    assert ("method", ("cpm", "bal"), ("leiden", "bal")) in pairs
    assert not comparison_pairs(
        ["leiden"], ["base"], ComparisonsConfig(pairs=(), reference_method=None)
    )


def test_descriptions(
    corpus: Corpus, edges: pd.DataFrame, clustering: dict[str, int], documents: dict[str, str]
) -> None:
    pairs = pd.DataFrame(
        {
            "query": [f"q{i // 2}" for i in range(len(documents))],
            "doc_id": list(documents),
            "is_selected": [i % 2 == 0 for i in range(len(documents))],
        }
    )
    description = collection_description(pairs, corpus, ["a", "b", "c"], ["d"])
    assert description["collection"]["documents"] == corpus.n_docs
    assert description["collection"]["judged_relevant_documents"] == corpus.n_docs // 2
    assert description["collection"]["queries"] == 4
    assert description["vocabulary"]["terms"] == len(corpus.vocabulary)
    assert len(description["document_frequency"]["top10"]) == 10

    graph = graph_description(edges, corpus, frame_stop_words=3)
    assert graph["terms"] == len(clustering) == graph["terms_in_corpus"]
    assert graph["components"] >= 1

    table = clusters_table(edges, clustering, top_clusters=2, top_terms=3)
    assert table["terms"].tolist() == sorted(table["terms"], reverse=True)
    assert all(len(terms.split()) == 3 for terms in table["top_terms"])
    assert ((table["density"] > 0) & (table["density"] <= 1)).all()


def test_frame_word_candidates(corpus: Corpus, tokenizer: Tokenizer) -> None:
    queries = ["define apple", "define dog", "apple juice", "what is the ocean"]
    table = frame_word_candidates(queries, corpus, tokenizer, listed={"define"}, top_n=3)
    define = table.set_index("term").loc["define"]
    assert define["query_share"] == 0.5
    assert define["document_share"] == 0
    assert define["lead_share"] == 1.0
    assert define["listed"]
    assert table["ratio"].is_monotonic_decreasing


def test_every_figure_renders(
    edges: pd.DataFrame, clustering: dict[str, int], tmp_path: Path
) -> None:
    strength = node_strength(edges)
    fanouts = {"hash_bal": np.array([3, 4, 2, 3]), "bal": np.array([1, 2, 2, 1])}
    curve = {"1": 0.5, "2": 0.8, "full": 1.0}
    points = pd.DataFrame(
        {
            "strategy": ["hash_bal", "bal"],
            "duplication": [23.0, 19.0],
            "fanout_mean": [3.0, 1.5],
            "overlap": [0.6, 0.8],
        }
    )
    by_sample = pd.DataFrame(
        {"train": [3.0, 1.5], "holdout": [3.1, 1.6]}, index=["hash_bal", "bal"]
    )
    retrieval = pd.DataFrame(
        {
            "method": ["leiden", "leiden", "cpm", "cpm"],
            "strategy": ["hash_bal", "bal"] * 2,
            "overlap": [0.6, 0.8, 0.6, 0.7],
            "overlap_ci_low": [0.55, 0.75, 0.55, 0.65],
            "overlap_ci_high": [0.65, 0.85, 0.65, 0.75],
        }
    )
    plots.plot_fanout_ecdf(fanouts, "t", tmp_path / "a.pdf")
    plots.plot_overlap_by_budget({"hash_bal": curve, "bal": curve}, 10, "t", tmp_path / "b.pdf")
    plots.plot_duplication_vs_fanout(points, "t", tmp_path / "c.pdf")
    plots.plot_fanout_by_sample(by_sample, "t", tmp_path / "d.pdf")
    plots.plot_methods_comparison(retrieval, "t", tmp_path / "e.pdf")
    plots.plot_cluster_sizes(clustering, 3, "t", tmp_path / "f.pdf")
    plots.plot_cluster_wordclouds(clustering, strength, 4, 1, "t", tmp_path / "g.pdf")
    plots.plot_graph_clusters(edges, clustering, 50, 1, "t", tmp_path / "h.pdf")
    plots.plot_tsne(edges, clustering, strength, 3, 1, "t", tmp_path / "i.pdf")
    plots.plot_cluster_heatmap(edges, clustering, 3, "t", tmp_path / "j.pdf")
    for name in "abcdefghij":
        assert (tmp_path / f"{name}.pdf").stat().st_size > 0


@pytest.mark.parametrize("name", ["a", "b"])
def test_figures_are_byte_stable(name: str, tmp_path: Path) -> None:
    for run in ("first", "second"):
        plots.plot_fanout_ecdf({"bal": np.array([1, 2, 2])}, name, tmp_path / run / "x.pdf")
    assert (tmp_path / "first" / "x.pdf").read_bytes() == (
        tmp_path / "second" / "x.pdf"
    ).read_bytes()
