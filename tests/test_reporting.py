from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from sharded_index.config import ComparisonsConfig
from sharded_index.data.corpus import Corpus
from sharded_index.graph.cooccurrence import node_strength
from sharded_index.reporting import overview, plots, style
from sharded_index.reporting.describe import (
    clusters_table,
    collection_description,
    frame_word_candidates,
    graph_description,
)
from sharded_index.reporting.labels import Labels
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
    points = pd.DataFrame(
        {
            "strategy": ["hash_bal", "bal"],
            "duplication": [23.0, 19.0],
            "fanout_mean": [3.0, 1.5],
            "label": ["hash_bal\n@1 0.60", "bal\n@1 0.80"],
        }
    )
    text = Labels("ru")
    plots.plot_duplication_vs_fanout(points, text, tmp_path / "a.pdf")
    plots.plot_cluster_sizes(clustering, 3, text, tmp_path / "b.pdf")
    plots.plot_cluster_wordclouds(clustering, strength, 4, 1, text, tmp_path / "c.pdf")
    plots.plot_graph_clusters(edges, clustering, 50, 1, tmp_path / "d.pdf")
    plots.plot_cluster_heatmap(edges, clustering, 3, text, tmp_path / "e.pdf")

    fanouts = {
        "leiden": {"bal": np.array([1, 2, 2, 1]), "hash_bal": np.array([3, 4, 2, 3])},
        "cpm": {"bal": np.array([1, 2, 3, 1]), "hash_bal": np.array([3, 3, 2, 3])},
    }
    overview.plot_fanout_ecdf(fanouts, text, tmp_path / "f.pdf")
    routing = pd.DataFrame(
        {
            "method": ["leiden"] * 4 + ["cpm"] * 4,
            "strategy": ["bal", "bal", "hash_bal", "hash_bal"] * 2,
            "sample": ["train", "holdout"] * 4,
            "single_shard_share": [0.1, 0.09, 0.06, 0.07, 0.1, 0.08, 0.06, 0.07],
            "single_shard_ci_low": [0.09, 0.08, 0.05, 0.06, 0.09, 0.07, 0.05, 0.06],
            "single_shard_ci_high": [0.11, 0.1, 0.07, 0.08, 0.11, 0.09, 0.07, 0.08],
            "fanout_mean": [2.8, 2.9, 3.2, 3.2, 2.9, 2.9, 3.2, 3.2],
            "fanout_ci_low": [2.7, 2.8, 3.1, 3.1, 2.8, 2.8, 3.1, 3.1],
            "fanout_ci_high": [2.9, 3.0, 3.3, 3.3, 3.0, 3.0, 3.3, 3.3],
        }
    )
    overview.plot_by_sample(
        routing,
        ["bal"],
        ["train", "holdout"],
        {"train": 10, "holdout": 6},
        text,
        tmp_path / "g.pdf",
    )
    sensitivity = pd.DataFrame(
        {
            "sweep": ["resolution"] * 4 + ["n_parts"] * 4 + ["seed"] * 2 + ["train_size"] * 4,
            "method": ["leiden"] * 4 + ["metis"] * 4 + ["leiden"] * 2 + ["infomap"] * 4,
            "value": [1.0, 2.0, 1.0, 2.0, 4.0, 8.0, 4.0, 8.0, 1.0, 1.0, 10.0, 40.0, 10.0, 40.0],
            "strategy": ["aff", "aff", "bal", "bal"] * 2
            + ["aff", "bal"]
            + ["aff", "aff", "bal", "bal"],
            "largest_shard": [
                0.8,
                0.6,
                0.15,
                0.15,
                0.3,
                0.25,
                0.15,
                0.15,
                0.8,
                0.15,
                0.4,
                0.36,
                0.15,
                0.15,
            ],
            "duplication": [
                10.0,
                14.0,
                20.0,
                20.0,
                21.0,
                22.0,
                20.0,
                20.0,
                10.0,
                20.0,
                20.0,
                20.6,
                20.0,
                20.1,
            ],
            "overlap_1": [
                0.9,
                0.85,
                0.78,
                0.77,
                0.7,
                0.72,
                0.74,
                0.75,
                0.9,
                0.78,
                0.74,
                0.75,
                0.77,
                0.77,
            ],
        }
    )
    overview.plot_parameter_sweeps(
        sensitivity, ["aff", "bal"], 0.15, 1, 10, text, tmp_path / "h.pdf"
    )
    ablations = pd.DataFrame(
        {
            "variant": ["baseline", "baseline", "v1", "v1"] * 2,
            "method": ["leiden", "cpm"] * 4,
            "strategy": ["bal"] * 4 + ["aff"] * 4,
            "overlap_1": [0.78, 0.76, 0.79, 0.75, 0.92, 0.73, 0.9, 0.74],
            "largest_shard": [0.15] * 4 + [0.83, 0.32, 0.8, 0.3],
        }
    )
    overview.plot_variant_heatmap(ablations, "bal", "overlap_1", "delta", "Δ", tmp_path / "i.pdf")
    overview.plot_variant_heatmap(
        ablations, "aff", "largest_shard", "value", "%", tmp_path / "j.pdf"
    )
    for name in "abcdefghij":
        assert (tmp_path / f"{name}.pdf").stat().st_size > 0


def test_style_codes_identity_consistently() -> None:
    colors = style.method_colors(["metis", "leiden", "custom"])
    assert colors["leiden"] == style.BLUE
    assert colors["metis"] == style.PURPLE
    assert colors["custom"] in style.SPARE_COLORS
    markers = style.method_markers(["cpm", "custom"])
    assert markers["cpm"] == "^"
    assert markers["custom"] in style.SPARE_MARKERS
    assert style.strategy_color("hash_bal") == style.HASH_COLOR
    assert style.strategy_color("base_r3") != style.strategy_color("base")
    assert style.strategy_marker("aff_r3") == style.strategy_marker("aff") == "s"
    assert style.marker_style("hash_aff", "#000000")["markerfacecolor"] == style.SURFACE
    assert style.strategy_line_style("hash_base") == "--"
    assert style.strategy_line_style("base") == "-"


def test_labels_exist_in_both_languages() -> None:
    russian, english = Labels("ru"), Labels("en")
    assert (
        russian("overlap_at", k=10, shards=russian.shards(1))
        == "Overlap@10 при опросе одного шарда"
    )
    assert (
        english("overlap_at", k=10, shards=english.shards(2)) == "Overlap@10 with 2 shards probed"
    )
    assert russian.sample("frequent_pairs") == english.sample("frequent_pairs") == "frequent pairs"
    assert english.sample("unknown_sample") == "unknown_sample"
    assert russian.number(241185) == "241\u2009185"
    assert english.number(241185) == "241,185"
    assert russian.panel(0) == "а)"
    assert english.panel(1) == "b)"


def test_point_labels_do_not_overlap_and_stay_inside_the_axes() -> None:
    figure = Figure(figsize=(6, 4), layout="tight")
    axis = figure.subplots()
    points = [(3.0, 3.0), (3.0, 3.0), (3.02, 3.01), (3.0, 3.02), (1.0, 1.0), (5.0, 5.0)]
    axis.scatter([x for x, _ in points], [y for _, y in points])
    labels = [f"p{i}\noverlap 0.5" for i in range(len(points))]
    boxes = plots._label_points(axis, figure, points, labels, ["k"] * len(points))
    assert len(boxes) == len(points)
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    inside = axis.bbox
    assert all(
        inside.x0 <= box.x0 and box.x1 <= inside.x1 and inside.y0 <= box.y0 and box.y1 <= inside.y1
        for box in boxes
    )


@pytest.mark.parametrize("name", ["a", "b"])
def test_figures_are_byte_stable(name: str, tmp_path: Path) -> None:
    points = pd.DataFrame(
        {"strategy": ["bal"], "duplication": [19.0], "fanout_mean": [2.9], "label": [name]}
    )
    for run in ("first", "second"):
        plots.plot_duplication_vs_fanout(points, Labels("en"), tmp_path / run / "x.pdf")
    assert (tmp_path / "first" / "x.pdf").read_bytes() == (
        tmp_path / "second" / "x.pdf"
    ).read_bytes()
