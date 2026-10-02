from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sharded_index.config import Bm25Config
from sharded_index.data.corpus import Corpus
from sharded_index.evaluation.membership import assign_documents
from sharded_index.index.bm25 import GlobalBM25
from sharded_index.index.build import build_index, build_shard_indices, corpus_fingerprint
from sharded_index.index.ranking import Rankings, ReferenceIndex, ranking_key, update_rankings
from sharded_index.index.search import ShardSearcher
from sharded_index.partition.model import TermPartition
from sharded_index.text import Tokenizer
from synthetic import QUERIES, STOP_WORDS

BM25 = Bm25Config(k1=1.2, b=0.75)


@pytest.fixture(scope="module")
def reference(
    tmp_path_factory: pytest.TempPathFactory, documents: dict[str, str], corpus: Corpus
) -> ReferenceIndex:
    index_dir = tmp_path_factory.mktemp("index")
    build_index(documents, index_dir, STOP_WORDS)
    return ReferenceIndex(index_dir, corpus.doc_ids, BM25)


def test_index_finds_exactly_the_documents_sharing_a_term(
    reference: ReferenceIndex, documents: dict[str, str], tokenizer: Tokenizer
) -> None:
    doc_tokens = [set(tokenizer(text)) for text in documents.values()]
    for query in QUERIES:
        ranking, scores = reference.search(query)
        expected = {i for i, tokens in enumerate(doc_tokens) if tokens & set(tokenizer(query))}
        assert set(ranking.tolist()) == expected
        assert np.all(np.diff(scores) <= 1e-12)


def test_limited_ranking_is_the_head_of_the_full_one(reference: ReferenceIndex) -> None:
    for query in QUERIES:
        full, _ = reference.search(query)
        head, _ = reference.search(query, limit=3)
        assert head.tolist() == full[:3].tolist()


def test_shards_score_documents_as_the_full_index_does(
    reference: ReferenceIndex,
    strategies: dict[str, TermPartition],
    corpus: Corpus,
    documents: dict[str, str],
    tokenizer: Tokenizer,
    tmp_path: Path,
) -> None:
    partition = strategies["base"]
    by_shard = assign_documents(partition, corpus).tocsc()
    shards = {
        shard: {
            corpus.doc_ids[doc]: documents[corpus.doc_ids[doc]]
            for doc in by_shard.indices[by_shard.indptr[shard] : by_shard.indptr[shard + 1]]
        }
        for shard in range(by_shard.shape[1])
        if by_shard.indptr[shard] < by_shard.indptr[shard + 1]
    }
    build_shard_indices(shards, tmp_path, STOP_WORDS, n_workers=1)
    searcher = ShardSearcher(tmp_path, GlobalBM25(reference.searcher, b=BM25.b, k1=BM25.k1))

    compared = 0
    for query in QUERIES:
        ranking, scores = reference.search(query)
        full_score = dict(zip((corpus.doc_ids[doc] for doc in ranking), scores, strict=True))
        for shard in partition.shards_of(tokenizer(query)):
            for doc_id, score in searcher.scores(shard, reference.parse(query)).items():
                assert score == pytest.approx(full_score[doc_id], abs=1e-9)
                compared += 1
    assert compared > 0


def test_fingerprint_depends_on_documents_and_stop_words() -> None:
    assert corpus_fingerprint(["b", "a"], {"the"}) == corpus_fingerprint(["a", "b"], {"the"})
    assert corpus_fingerprint(["a", "b"], {"the"}) != corpus_fingerprint(["a"], {"the"})
    assert corpus_fingerprint(["a", "b"], {"the"}) != corpus_fingerprint(["a", "b"], {"a"})


def test_rankings_roundtrip_and_evaluated_mask(reference: ReferenceIndex, tmp_path: Path) -> None:
    arrays = [reference.search(query, limit=3)[0] for query in QUERIES]
    truncated = np.array([len(array) == 3 for array in arrays])
    Rankings(QUERIES, arrays, truncated).save(tmp_path / "r.npz", "key")
    loaded = Rankings.load(tmp_path / "r.npz")
    assert loaded.queries == QUERIES
    assert [array.tolist() for array in loaded.arrays] == [array.tolist() for array in arrays]
    assert loaded.evaluated.tolist() == [True, True, True, True, False]
    assert Rankings.key_of(tmp_path / "r.npz") == "key"


def test_known_queries_are_not_ranked_again(
    reference: ReferenceIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    searched: list[str] = []
    original = reference.search

    def counting(query: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        searched.append(query)
        return original(query, limit)

    monkeypatch.setattr(reference, "search", counting)
    key = ranking_key("fingerprint", 3, BM25)
    update_rankings(tmp_path, {"first": QUERIES[:3], "stale": QUERIES[:1]}, reference, 3, key)
    assert searched == QUERIES[:3]

    update_rankings(tmp_path, {"first": QUERIES[:3], "second": QUERIES[1:]}, reference, 3, key)
    assert searched == QUERIES
    assert sorted(path.stem for path in tmp_path.glob("*.npz")) == ["first", "second"]
    assert Rankings.load(tmp_path / "second.npz").queries == QUERIES[1:]

    update_rankings(tmp_path, {"first": QUERIES[:3]}, reference, 3, ranking_key("other", 3, BM25))
    assert searched == QUERIES + QUERIES[:3]
