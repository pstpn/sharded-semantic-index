from __future__ import annotations

import numpy as np
import pandas as pd

from sharded_index.data.corpus import Corpus
from sharded_index.evaluation.verification import cover_is_complete
from sharded_index.partition.model import TermPartition
from sharded_index.routing import Router, cover_weights, greedy_cover
from sharded_index.text import Tokenizer
from synthetic import QUERIES


def test_greedy_cover_takes_the_heaviest_shard_first() -> None:
    candidates = {"a": (0,), "b": (1,), "c": (1,)}
    assert greedy_cover(candidates, {"a": 5.0, "b": 1.0, "c": 1.0}) == [0, 1]
    assert greedy_cover(candidates, {"a": 1.0, "b": 1.0, "c": 1.0}) == [1, 0]
    assert greedy_cover(candidates, {"a": 1.0, "b": 0.5, "c": 0.5}) == [0, 1]


def test_greedy_cover_prefers_a_shard_holding_several_terms() -> None:
    candidates = {"a": (0, 2), "b": (1, 2), "c": (3,)}
    weights = {"a": 1.0, "b": 1.0, "c": 1.5}
    assert greedy_cover(candidates, weights) == [2, 3]
    assert greedy_cover(candidates, weights, max_shards=1) == [2]
    assert greedy_cover({}, weights) == []


def test_cover_weights(corpus: Corpus, edges: pd.DataFrame) -> None:
    idf = cover_weights("idf", corpus, edges)
    assert set(idf) == set(corpus.vocabulary)
    rare, frequent = np.argmin(corpus.df), np.argmax(corpus.df)
    assert idf[corpus.vocabulary[rare]] > idf[corpus.vocabulary[frequent]]
    strength = cover_weights("strength", corpus, edges)
    assert "windows10" not in strength
    assert strength["apple"] > 0


def test_router_probes_the_heaviest_term_first(
    strategies: dict[str, TermPartition], corpus: Corpus, tokenizer: Tokenizer
) -> None:
    partition = strategies["base"]
    weights = {**dict.fromkeys(corpus.vocabulary, 1.0), "windows10": 100.0}
    cover = Router(partition, weights, tokenizer).cover("python bug windows10")
    assert cover[0] == partition.primary["windows10"]


def test_covers_are_complete_and_replication_never_grows_them(
    strategies: dict[str, TermPartition],
    corpus: Corpus,
    edges: pd.DataFrame,
    tokenizer: Tokenizer,
) -> None:
    weights = cover_weights("idf", corpus, edges)
    single = Router(strategies["base"], weights, tokenizer)
    replicated = Router(strategies["base_r3"], weights, tokenizer)
    for query in QUERIES:
        cover = single.cover(query)
        assert cover_is_complete(single, query, cover)
        assert set(cover) == strategies["base"].shards_of(tokenizer(query))
        if len(cover) > 1:
            assert not cover_is_complete(single, query, cover[:1])
        assert cover_is_complete(replicated, query, replicated.cover(query))
        assert len(replicated.cover(query)) <= len(cover)
