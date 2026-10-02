from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from whoosh import scoring

from sharded_index.config import Bm25Config, Strategy
from sharded_index.data.corpus import Corpus
from sharded_index.evaluation import verification
from sharded_index.evaluation.measure import (
    budget_labels,
    expected_hash_duplication,
    measure_queries,
    partition_stats,
)
from sharded_index.evaluation.membership import (
    assign_by_lookup,
    assign_documents,
    same_membership,
)
from sharded_index.evaluation.slices import (
    OUT_OF_GRAPH,
    SINGLE_TERM,
    connectivity_slices,
    term_count_slices,
)
from sharded_index.evaluation.statistics import (
    bootstrap_ci,
    bootstrap_weights,
    retrieval_summary,
    routing_summary,
)
from sharded_index.evaluation.verification import (
    PhysicalCheck,
    check_completeness,
    check_reference,
    reference_results,
)
from sharded_index.index.build import build_index
from sharded_index.index.ranking import Rankings, ReferenceIndex
from sharded_index.partition.hashing import id_space
from sharded_index.partition.model import TermPartition
from sharded_index.routing import Router, cover_weights
from sharded_index.text import Tokenizer
from synthetic import QUERIES, STOP_WORDS

BM25 = Bm25Config(k1=1.2, b=0.75)
BUDGETS = (1, 2)
TOP_K = 3


@pytest.fixture(scope="module")
def reference(
    tmp_path_factory: pytest.TempPathFactory, documents: dict[str, str], corpus: Corpus
) -> ReferenceIndex:
    index_dir = tmp_path_factory.mktemp("index")
    build_index(documents, index_dir, STOP_WORDS)
    return ReferenceIndex(index_dir, corpus.doc_ids, BM25)


@pytest.fixture(scope="module")
def rankings(reference: ReferenceIndex) -> Rankings:
    arrays = [reference.search(query, limit=20)[0] for query in QUERIES]
    return Rankings(QUERIES, arrays, np.array([len(array) == 20 for array in arrays]))


@pytest.fixture(scope="module")
def router(
    strategies: dict[str, TermPartition], corpus: Corpus, edges: pd.DataFrame, tokenizer: Tokenizer
) -> Router:
    return Router(strategies["base"], cover_weights("idf", corpus, edges), tokenizer)


def test_matrix_assignment_equals_document_lookup(
    strategies: dict[str, TermPartition],
    corpus: Corpus,
    documents: dict[str, str],
    tokenizer: Tokenizer,
) -> None:
    token_sets = [set(tokenizer(text)) for text in documents.values()]
    for partition in strategies.values():
        membership = assign_documents(partition, corpus)
        assert same_membership(membership, assign_by_lookup(partition, token_sets))
    broken = membership.tolil()
    broken[0, :] = False
    assert not same_membership(broken.tocsr(), membership)


def test_measurements_of_a_query_sample(
    router: Router, corpus: Corpus, rankings: Rankings, clustering: dict[str, int]
) -> None:
    membership = assign_documents(router.partition, corpus)
    frame = measure_queries(router, membership, rankings, BUDGETS, TOP_K, id_space(clustering))
    measured, unmatched = frame.iloc[:-1], frame.iloc[-1]

    assert frame["evaluated"].tolist() == [True, True, True, True, False]
    assert frame["terms"].tolist() == [3, 3, 3, 2, 3]
    assert frame["known_terms"].tolist() == [3, 3, 3, 2, 0]
    assert unmatched[["fanout", "overlap_full", "volume_1"]].isna().all()
    assert unmatched["first_shard"] == -1
    assert measured["fanout"].tolist() == [len(router.cover(query)) for query in QUERIES[:-1]]
    assert (measured["overlap_full"] == 1.0).all()
    assert (measured["overlap_1"] <= measured["overlap_2"]).all()
    assert (measured["overlap_2"] <= measured["overlap_full"]).all()
    assert (measured["volume_1"] <= measured["volume_2"]).all()
    assert measured["hash_probes"].tolist() == [0, 0, 1, 0]

    sizes = np.asarray(membership.sum(axis=0)).ravel() / corpus.n_docs
    for row, query in zip(measured.itertuples(), QUERIES[:-1], strict=True):
        cover = router.cover(query)
        assert row.first_shard == cover[0]
        assert row.volume_1 == pytest.approx(sizes[cover[0]])
        assert row.volume_full == pytest.approx(sizes[cover].sum())
        in_first = membership[rankings.arrays[row.position][:TOP_K]][:, cover[0]].toarray()
        assert row.overlap_1 == pytest.approx(in_first.mean())


def test_summaries_ignore_unevaluated_queries(
    router: Router, corpus: Corpus, rankings: Rankings, clustering: dict[str, int]
) -> None:
    membership = assign_documents(router.partition, corpus)
    frame = measure_queries(router, membership, rankings, BUDGETS, TOP_K, id_space(clustering))
    routing = routing_summary(frame)
    covers = [len(router.cover(query)) for query in QUERIES[:-1]]
    assert (routing["queries"], routing["evaluated"]) == (5, 4)
    assert routing["terms_mean"] == routing["known_terms_mean"] == 2.75
    assert routing["fanout_mean"] == np.mean(covers)
    assert routing["single_shard_share"] == np.mean([size == 1 for size in covers])
    assert routing["mixed_query_share"] == 0.25

    retrieval = retrieval_summary(frame, BUDGETS)
    assert list(retrieval) == budget_labels(BUDGETS) == ["1", "2", "full"]
    assert retrieval["full"]["overlap"] == 1.0
    assert retrieval["1"]["overlap"] <= retrieval["2"]["overlap"]


def test_partition_statistics(strategies: dict[str, TermPartition], corpus: Corpus) -> None:
    lengths = np.asarray(corpus.doc_terms.sum(axis=1)).ravel()
    stats = {}
    for name, partition in strategies.items():
        membership = assign_documents(partition, corpus)
        stats[name] = partition_stats(partition, membership, corpus, Strategy.parse(name), None)
        assert stats[name]["duplication"] == membership.nnz / corpus.n_docs
        assert 1.0 <= stats[name]["duplication"] <= lengths.max()
        assert 0 < stats[name]["largest_shard"] <= 1

    assert (
        stats["base_r3"]["duplication"]
        >= stats["base_r2"]["duplication"]
        >= stats["base"]["duplication"]
    )
    assert stats["base"]["replicated_terms"] == 0
    assert stats["base_r2"]["replicated_terms"] > 0
    assert 0 < stats["base_r2"]["replica_cost_top10pct"] <= 1
    assert np.isnan(stats["base"]["expected_duplication"])
    n_shards = int(stats["hash_bal"]["shards"])
    assert stats["hash_bal"]["expected_duplication"] == expected_hash_duplication(corpus, n_shards)
    assert stats["hash_bal"]["expected_duplication"] == pytest.approx(
        stats["hash_bal"]["duplication"], rel=0.15
    )


def test_bootstrap_matches_explicit_resampling() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(60, 2))
    values[::7, 0] = np.nan
    weights = bootstrap_weights(60, 200, seed=5)
    low, high = bootstrap_ci(values, weights, 0.95)

    draws = np.random.default_rng(5).integers(0, 60, size=(200, 60))
    means = np.nanmean(values[draws], axis=1)
    assert low == pytest.approx(np.percentile(means, 2.5, axis=0))
    assert high == pytest.approx(np.percentile(means, 97.5, axis=0))
    assert (low < np.nanmean(values, axis=0)).all()
    assert (np.nanmean(values, axis=0) < high).all()


def test_connectivity_slices(edges: pd.DataFrame, tokenizer: Tokenizer) -> None:
    strongest = edges.iloc[0]
    queries = [
        f"{strongest['src']} {strongest['dst']}",
        "apple dog ocean",
        "apple",
        "apple windows10",
        "ignored query",
    ]
    slices = connectivity_slices(queries, [True, True, True, True, False], edges, tokenizer, 2)
    assert (slices["slicing"] == "connectivity").all()
    by_position = slices.set_index("position")
    assert by_position["slice"].to_dict() == {0: "Q2", 1: "Q1", 2: SINGLE_TERM, 3: OUT_OF_GRAPH}
    assert by_position.loc[0, "connectivity"] == pytest.approx(strongest["weight"])
    assert by_position.loc[[2, 3], "connectivity"].isna().all()


def test_term_count_slices() -> None:
    slices = term_count_slices([2, 1, 7, 3, 3, 4], [True, True, True, True, False, True], 3)
    assert (slices["slicing"] == "terms").all()
    assert slices["slice"].tolist() == ["1", "2", "3+", "3+", "3+"]
    assert slices["position"].tolist() == [1, 0, 2, 3, 5]


def test_verification_passes_and_detects_broken_inputs(
    router: Router,
    corpus: Corpus,
    documents: dict[str, str],
    tokenizer: Tokenizer,
    reference: ReferenceIndex,
    rankings: Rankings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = router.partition
    membership = assign_documents(partition, corpus)
    token_sets = [set(tokenizer(text)) for text in documents.values()]
    samples = {"sample": QUERIES[:-1]}

    completeness = check_completeness(partition, membership, token_sets, router, samples)
    assert completeness == {
        "assignment_exact": True,
        "incomplete_covers": 0,
        "incomplete_covers_by_sample": {"sample": 0},
        "queries_checked": 4,
    }
    broken = membership.tolil()
    broken[0, :] = False
    assert not check_completeness(partition, broken.tocsr(), token_sets, router, samples)[
        "assignment_exact"
    ]

    references = reference_results(reference, QUERIES[:-1], corpus.doc_ids)
    saved = dict(zip(rankings.queries, rankings.arrays, strict=True))
    assert check_reference(references, saved, 1e-9) == {
        "ranking_order_violations": 0,
        "ranking_prefix_mismatches": 0,
    }
    reversed_rankings = {query: array[::-1] for query, array in saved.items()}
    assert check_reference(references, reversed_rankings, 1e-9)["ranking_prefix_mismatches"] > 0

    check = PhysicalCheck(
        reference, references, documents, corpus.doc_ids, BUDGETS, BM25, 1e-9, workers=1
    )
    physical = check.run(membership, router, tmp_path / "shards")
    assert physical["scores_compared"] > 0
    assert physical["max_score_difference"] <= 1e-9
    assert physical["result_mismatches"] == 0
    assert not (tmp_path / "shards").exists()

    # With the statistics of the shard itself the scores differ: the check must notice.
    monkeypatch.setattr(
        verification, "GlobalBM25", lambda _reference, b, k1: scoring.BM25F(B=b, K1=k1)
    )
    local = check.run(membership, router, tmp_path / "shards")
    assert local["max_score_difference"] > 1e-9
    assert local["result_mismatches"] > 0
