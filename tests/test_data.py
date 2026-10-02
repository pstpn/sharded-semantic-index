from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sharded_index.config import QueriesConfig, SyntheticConfig
from sharded_index.data.corpus import Corpus
from sharded_index.data.msmarco import extract_pairs, split_queries
from sharded_index.data.queries import build_samples, queries_of, sample_queries, synthetic_queries
from sharded_index.storage import load_incidence, save_incidence
from sharded_index.text import Tokenizer

ROWS = [
    {
        "query": "Make apple juice",
        "passages": {
            "passage_text": ["Irrelevant one.", "Squeeze THE apples.", "Another.", "Also good."],
            "is_selected": [0, 1, 0, 1],
        },
    },
    {
        "query": "one judged",
        "passages": {
            "passage_text": ["distractor a", "the answer", "distractor a", "distractor b"],
            "is_selected": [0, 1, 0, 0],
        },
    },
    {"query": "none selected", "passages": {"passage_text": ["a b", "c d"], "is_selected": [0, 0]}},
    {"query": "", "passages": {"passage_text": ["x"], "is_selected": [1]}},
]


def test_extraction_takes_judged_passages_first() -> None:
    pairs = extract_pairs(ROWS, max_rows=10, docs_per_query=2)
    by_query = {
        query: list(zip(group["doc_text"], group["is_selected"], strict=True))
        for query, group in pairs.groupby("query", sort=False)
    }
    assert by_query == {
        "make apple juice": [("squeeze the apples", True), ("also good", True)],
        "one judged": [("the answer", True), ("distractor a", False)],
    }


def test_extraction_respects_max_rows_and_hashes_documents() -> None:
    pairs = extract_pairs(ROWS, max_rows=1, docs_per_query=2)
    assert pairs["query"].unique().tolist() == ["make apple juice"]
    assert pairs["doc_id"].str.fullmatch(r"[0-9a-f]{32}").all()


def test_split_keeps_dataset_order() -> None:
    pairs = pd.DataFrame({"query": ["a", "a", "b", "c", "d", "e"]})
    assert split_queries(pairs, 0.6) == (["a", "b", "c"], ["d", "e"])


def test_corpus_incidence_and_document_frequency(tokenizer: Tokenizer) -> None:
    corpus = Corpus.from_documents({"x": "apple the juice", "y": "juice juice dog"}, tokenizer)
    assert corpus.doc_ids == ["x", "y"]
    assert corpus.vocabulary == ["apple", "dog", "juice"]
    assert corpus.doc_terms.toarray().tolist() == [[1, 0, 1], [0, 1, 1]]
    assert corpus.df.tolist() == [1, 1, 2]


def test_corpus_roundtrip_is_lossless_and_byte_stable(corpus: Corpus, tmp_path: Path) -> None:
    corpus.save(tmp_path / "first")
    corpus.save(tmp_path / "second")
    loaded = Corpus.load(tmp_path / "first")
    assert loaded.doc_ids == corpus.doc_ids
    assert loaded.vocabulary == corpus.vocabulary
    assert (loaded.doc_terms != corpus.doc_terms).nnz == 0
    for name in ("doc_terms.npz", "vocabulary.parquet", "documents.parquet"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()


def test_incidence_storage_keeps_shape_with_empty_columns(tmp_path: Path) -> None:
    matrix = np.zeros((3, 5), dtype=bool)
    matrix[0, 1] = matrix[2, 1] = matrix[2, 3] = True
    save_incidence(tmp_path / "m.npz", matrix)
    assert load_incidence(tmp_path / "m.npz").toarray().tolist() == matrix.tolist()


def test_sample_is_seeded_and_ordered() -> None:
    pool = [f"q{i}" for i in range(50)]
    sample = sample_queries(pool, 10, seed=1)
    assert len(sample) == 10
    assert sample == sorted(sample, key=pool.index)
    assert sample == sample_queries(pool, 10, seed=1)
    assert sample != sample_queries(pool, 10, seed=2)
    assert sample_queries(pool, 100, seed=1) == pool
    assert sample_queries(pool, None, seed=1) == pool


def test_synthetic_pairs_are_the_top_edges(edges: pd.DataFrame) -> None:
    connected = synthetic_queries(edges, "connected", 2, 5)
    assert connected[0].split() == edges.iloc[0][["src", "dst"]].tolist()
    top = edges[edges["count"] == edges["count"].max()]
    most_frequent = {frozenset(pair) for pair in zip(top["src"], top["dst"], strict=True)}
    assert frozenset(synthetic_queries(edges, "frequent", 2, 5)[0].split()) in most_frequent


def test_synthetic_triples_are_distinct_term_sets(edges: pd.DataFrame) -> None:
    for kind in ("connected", "frequent"):
        triples = synthetic_queries(edges, kind, 3, 8)
        assert triples
        assert all(len(set(triple.split())) == 3 for triple in triples)
        assert len({frozenset(triple.split()) for triple in triples}) == len(triples)


def test_samples_table_lists_every_sample(edges: pd.DataFrame) -> None:
    config = QueriesConfig(
        train_ratio=0.8,
        sample_seed=1,
        train_size=3,
        holdout_size=None,
        ood=None,
        synthetic=SyntheticConfig(size=4, kinds=("connected", "frequent"), arities=(2, 3)),
    )
    train, holdout = [f"t{i}" for i in range(10)], ["h0", "h1"]
    samples = build_samples(train, holdout, edges, config)
    assert list(dict.fromkeys(samples["sample"])) == config.sample_names()
    assert len(queries_of(samples, "train")) == 3
    assert queries_of(samples, "holdout") == holdout
    assert len(queries_of(samples, "frequent_pairs")) == 4
