"""End-to-end smoke test of the library on a synthetic corpus.

Run from the repo root:  python tests/smoke_test.py

Checks the pipeline invariants rather than exact numbers:
- partitions cover the vocabulary, fallback space is disjoint from the core;
- save/load roundtrip is lossless;
- full-fanout routing recall is exactly 1.0 — documents are replicated to all
  shards of their terms, so any ground-truth match is always reachable;
- recall grows monotonically with fanout;
- all figures render headlessly.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import numpy as np
from sharded_index import (
    PartitionConfig,
    ReplicatedTermPartition,
    ShardedSearcher,
    TermPartition,
    assign_docs_to_shards,
    build_whoosh_indices,
    extract_query_doc_pairs,
    index_size_mb,
    load_partition,
    normalize_text,
    replicate_with_volume_guard,
    tokenize,
)
from sharded_index.clusters import build_term_graph, cluster_summary_table, top_terms_per_cluster
from sharded_index.evaluation import (
    build_ground_truth,
    build_ground_truth_fast,
    compute_routing_recall,
    recall_by_fanout,
    term_postings,
)
from sharded_index.plots import (
    plot_cluster_size_distribution,
    plot_cluster_wordclouds,
    plot_fanout_distribution,
    plot_graph_clusters,
    plot_inter_cluster_heatmap,
    plot_recall_vs_fanout,
    plot_tsne_projection,
)

TOPICS = [
    ["apple", "banana", "fruit", "juice", "sweet"],
    ["dog", "cat", "pet", "animal", "fur"],
    ["python", "code", "bug", "program", "compile"],
    ["ocean", "wave", "beach", "sand", "swim"],
]


def make_corpus(n: int = 400) -> list[str]:
    rng = np.random.default_rng(0)
    texts = []
    for i in range(n):
        words = list(rng.choice(TOPICS[i % 4], size=rng.integers(2, 5), replace=False))
        if i % 7 == 0:
            words.append("the")        # stop word — must vanish everywhere
        if i % 5 == 0:
            words.append("windows10")  # digit token — routed, but never in the graph
        rng.shuffle(words)
        texts.append(" ".join(words))
    return texts


def main() -> None:
    texts = make_corpus()
    docs = {f"d{i:03d}": t for i, t in enumerate(texts)}

    # ── Partition construction ────────────────────────────────────────
    core = TermPartition.from_corpus(texts, config=PartitionConfig())
    assert core.n_shards > 1, "Leiden should find several clusters"
    assert "the" not in core.term_to_shard, "stop words must not enter the graph"
    assert "windows10" not in core.term_to_shard, "digit tokens must not enter the graph"

    corpus_terms = set()
    for text in docs.values():
        corpus_terms.update(tokenize(text))

    full_part = core.with_hash_fallback(corpus_terms)
    assert set(full_part.term_to_shard) == corpus_terms, "fallback must cover the vocabulary"
    fallback_shards = {s for t, s in full_part.term_to_shard.items() if t not in core.term_to_shard}
    assert all(s >= core.n_shards for s in fallback_shards), "fallback space must be disjoint"

    hash_part = TermPartition.from_hashing(corpus_terms, core.n_shards,
                                           node_strength=core.node_strength)
    assert set(hash_part.term_to_shard) == corpus_terms

    # ── save / load roundtrip ─────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        core.save(Path(tmp) / "part")
        loaded = TermPartition.load(Path(tmp) / "part")
        assert loaded.term_to_shard == core.term_to_shard
        assert loaded.node_strength == core.node_strength
        assert loaded.edges_df.equals(core.edges_df)
        assert loaded.clusters_df.equals(core.clusters_df)

    # ── Indices, search, evaluation ───────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        doc_shards, docs_by_shard = assign_docs_to_shards(docs, full_part)
        build_whoosh_indices(docs_by_shard, tmp / "sharded", overwrite=True)
        build_whoosh_indices(docs_by_shard, tmp / "sharded")  # cache path: must not rebuild
        build_whoosh_indices({0: docs}, tmp / "full", overwrite=True)
        assert index_size_mb(tmp / "sharded") > index_size_mb(tmp / "full") > 0

        searcher = ShardedSearcher(tmp / "sharded", full_part)
        hits = searcher.search("apple juice", top_k=5)
        assert hits and all("apple" in h.text or "juice" in h.text for h in hits)

        queries = ["apple juice sweet", "dog cat pet", "python bug windows10",
                   "ocean swim", "zzz nothing matches"]
        ground_truth = build_ground_truth(queries, tmp / "full" / "shard_000")

        # Контракт паритета токенизации: эталон в памяти (>=1 общий терм)
        # обязан в точности совпадать с небюджетным BM25-поиском Whoosh.
        doc_tokens_map = {doc_id: set(tokenize(text)) for doc_id, text in docs.items()}
        fast_gt = build_ground_truth_fast(queries, term_postings(doc_tokens_map))
        assert fast_gt == ground_truth, "fast GT must equal Whoosh GT (tokenizer parity)"

        routing = compute_routing_recall(queries, ground_truth, full_part, doc_shards)
        assert routing.recall == 1.0, (
            "full-fanout routing recall must be 1.0 by construction "
            f"(got {routing.recall})"
        )

        curve = recall_by_fanout(queries, ground_truth, full_part, doc_shards, max_fanout=4)
        values = [curve[f] for f in sorted(curve)]
        assert values == sorted(values), "recall must not decrease with fanout"

    # ── Term replication + cover routing ──────────────────────────────
    replicated = ReplicatedTermPartition.from_partition(full_part, 2)
    assert set(replicated.term_to_shards) == corpus_terms
    for term, shards in replicated.term_to_shards.items():
        assert shards[0] == full_part.term_to_shard[term], "primary shard must be preserved"
        assert len(shards) == len(set(shards)) <= 2
        if term in core.term_to_shard:
            assert all(s < core.n_shards for s in shards), "replicas must stay semantic"
        else:
            assert len(shards) == 1, "fallback terms are never replicated"
    assert 1.0 <= replicated.replication_factor <= 2.0

    queries = ["apple juice sweet", "dog cat pet", "python bug windows10",
               "ocean swim", "zzz nothing matches"]

    # For single assignment the greedy cover equals ranked routing (up to ties).
    for q in queries:
        cover = full_part.cover_shards_for_query(q)
        ranked = full_part.ranked_shards_for_query(q)
        assert set(cover) == set(ranked) and len(cover) == len(ranked), q

    # Replication may only shrink the cover, never grow it.
    cover_r1 = [len(full_part.cover_shards_for_query(q)) for q in queries]
    cover_r2 = [len(replicated.cover_shards_for_query(q)) for q in queries]
    assert np.mean(cover_r2) <= np.mean(cover_r1)

    # Affinity-quantile threshold: 0.0 keeps every candidate, 1.0 keeps only
    # the strongest — replication factor must shrink monotonically.
    no_threshold = ReplicatedTermPartition.from_partition(full_part, 2, affinity_quantile=0.0)
    strict = ReplicatedTermPartition.from_partition(full_part, 2, affinity_quantile=1.0)
    assert no_threshold.term_to_shards == replicated.term_to_shards
    assert strict.replication_factor <= replicated.replication_factor

    # ── Document-affinity fallback ────────────────────────────────────
    doc_token_sets = [set(tokenize(t)) for t in texts]
    aff = core.with_affinity_fallback(doc_token_sets)
    assert set(aff.term_to_shard) == corpus_terms, "affinity fallback must cover the vocabulary"
    for term, cluster in core.term_to_shard.items():
        if term in aff.term_to_shard:
            assert aff.term_to_shard[term] == cluster, "graph terms keep their cluster"
    # windows10 co-occurs with topic terms in documents → semantic space now
    assert aff.term_to_shard["windows10"] < core.n_shards
    # deterministic result (sorted construction, md5 residual)
    assert aff.term_to_shard == core.with_affinity_fallback(doc_token_sets).term_to_shard
    # a term that never meets a graph term still falls back to the hash space
    lonely = core.with_affinity_fallback([{"zzzalone"}])
    assert lonely.term_to_shard["zzzalone"] >= core.n_shards
    # mixed query (semantic + rare term) must not need MORE shards than with hash fallback
    for q in ["apple windows10", "dog noise5", "ocean windows10 swim"]:
        assert len(aff.shards_for_query(q)) <= len(full_part.shards_for_query(q)), q
    # replication composes with the affinity fallback
    aff_r2 = ReplicatedTermPartition.from_partition(aff, 2)
    assert aff_r2.replication_factor >= 1.0
    assert set(aff_r2.term_to_shards) == corpus_terms

    # ── Volume-capped splitting + guarded replication ─────────────────
    from collections import Counter

    term_df_syn: Counter[str] = Counter()
    for tokens in doc_token_sets:
        term_df_syn.update(tokens)
    cap_ratio = 0.15
    cap_kwargs = dict(
        max_volume_ratio=cap_ratio, resolution=1.0, seed=42,
        growth=2.0, max_rounds=5, fallback_offset=core.n_shards,
    )
    bal = aff.with_volume_cap(doc_token_sets, dict(term_df_syn), **cap_kwargs)
    assert set(bal.term_to_shard) == corpus_terms, "cap must preserve the vocabulary"
    assert bal.term_to_shard == aff.with_volume_cap(
        doc_token_sets, dict(term_df_syn), **cap_kwargs,
    ).term_to_shard, "volume cap must be deterministic"

    # Every shard fits the budget, up to the floor of its dominant term.
    vols: Counter[int] = Counter()
    for tokens in doc_token_sets:
        vols.update(bal.shards_for_tokens(tokens))
    cap_docs = cap_ratio * len(doc_token_sets)
    dominant_by_shard: dict[int, int] = {}
    for t, s in bal.term_to_shard.items():
        dominant_by_shard[s] = max(dominant_by_shard.get(s, 0), term_df_syn[t])
    for shard, vol in vols.items():
        assert vol <= max(cap_docs, dominant_by_shard[shard] * 1.05) + 1, (shard, vol)

    # Completeness invariant survives the cap: any doc sharing a term with
    # the query is reachable through the greedy cover.
    doc_shards_bal, _ = assign_docs_to_shards(docs, bal)
    for q in queries:
        q_tokens = set(tokenize(q))
        probed = set(bal.cover_shards_for_query(q))
        for doc_id, text in docs.items():
            if q_tokens & set(tokenize(text)):
                assert doc_shards_bal[doc_id] & probed, (q, doc_id)

    bal_r2 = replicate_with_volume_guard(
        bal, 2, doc_token_sets, dict(term_df_syn), max_volume_ratio=cap_ratio,
    )
    assert set(bal_r2.term_to_shards) == corpus_terms
    for term, shards in bal_r2.term_to_shards.items():
        assert shards[0] == bal.term_to_shard[term], "primary shard must be preserved"
    assert bal_r2.replication_factor >= 1.0
    doc_shards_bal_r2, _ = assign_docs_to_shards(docs, bal_r2)
    for q in queries:
        q_tokens = set(tokenize(q))
        probed = set(bal_r2.cover_shards_for_query(q))
        for doc_id, text in docs.items():
            if q_tokens & set(tokenize(text)):
                assert doc_shards_bal_r2[doc_id] & probed, (q, doc_id)

    # Full cover ⇒ recall 1.0 holds for replicated partitions too.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        doc_shards_r2, _ = assign_docs_to_shards(docs, replicated)
        dup_r1 = np.mean([len(s) for s in doc_shards.values()])
        dup_r2 = np.mean([len(s) for s in doc_shards_r2.values()])
        assert dup_r2 >= dup_r1, "replication must not reduce duplication"

        build_whoosh_indices({0: docs}, tmp / "full", overwrite=True)
        gt = build_ground_truth(queries, tmp / "full" / "shard_000")
        for q in queries:
            probed = set(replicated.cover_shards_for_query(q))
            for doc_id in gt[q]:
                assert doc_shards_r2[doc_id] & probed, (q, doc_id)

    # save / load / dispatch roundtrip
    with tempfile.TemporaryDirectory() as tmp:
        replicated.save(Path(tmp) / "r2")
        full_part.save(Path(tmp) / "r1")
        loaded_r2 = load_partition(Path(tmp) / "r2")
        loaded_r1 = load_partition(Path(tmp) / "r1")
        assert isinstance(loaded_r2, ReplicatedTermPartition)
        assert isinstance(loaded_r1, TermPartition)
        assert loaded_r2.term_to_shards == replicated.term_to_shards
        assert loaded_r2.node_strength == replicated.node_strength

    # ── MS MARCO extraction rules ─────────────────────────────────────
    rows = [
        {"query": "Make apple juice", "passages": {
            "passage_text": ["Squeeze THE apples.", "Irrelevant.", "Also good."],
            "is_selected": [1, 0, 1]}},
        {"query": "none selected", "passages": {
            "passage_text": ["a", "b"], "is_selected": [0, 0]}},
        {"query": "no flags", "passages": {"passage_text": ["plain passage"]}},
        {"query": "", "passages": {"passage_text": ["skipped: empty query"]}},
    ]
    strict = extract_query_doc_pairs(rows, selected_only=True)
    assert [p["query"] for p in strict].count(normalize_text("Make apple juice")) == 2
    assert not any(p["query"] == "selected" for p in strict), "row without selection is skipped"
    assert any(p["query"] == normalize_text("no flags") for p in strict)
    relaxed = extract_query_doc_pairs(rows, selected_only=False)
    assert any(p["query"] == normalize_text("none selected") for p in relaxed)

    # ── Analysis tables and figures (headless) ────────────────────────
    table = cluster_summary_table(core, top_n=5)
    assert {"cluster", "size", "density", "top_words"} <= set(table.columns)
    assert len(top_terms_per_cluster(core, n_clusters=3, n_terms=5)) == 3

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        graph = build_term_graph(core)
        fanouts = {"Hash": [3, 4, 2, 3], "Leiden": [1, 2, 2, 1]}
        plot_fanout_distribution(fanouts, save_path=tmp / "a.pdf")
        plot_recall_vs_fanout({"Hash": curve, "Leiden": curve}, {"Hash": 3.0, "Leiden": 1.5},
                              save_path=tmp / "b.pdf")
        plot_cluster_size_distribution(core.clusters_df, save_path=tmp / "c.pdf")
        plot_cluster_wordclouds(core, top_n=4, save_path=tmp / "d.pdf")
        plot_graph_clusters(graph, core, max_nodes=50, save_path=tmp / "e.pdf")
        plot_tsne_projection(graph, core, top_n_clusters=3, save_path=tmp / "f.pdf")
        plot_inter_cluster_heatmap(core, top_n_clusters=3, save_path=tmp / "g.pdf")
        for name in "abcdefg":
            assert (tmp / f"{name}.pdf").stat().st_size > 0

    print("smoke test: OK")


if __name__ == "__main__":
    main()
