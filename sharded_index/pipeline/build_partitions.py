"""Stage 2: term partitions.

- ``leiden_core`` — Leiden clusters of the NPMI graph built on **train** queries;
- ``hash`` — random baseline over the corpus vocabulary;
- ``leiden`` — core + hash fallback for out-of-graph terms;
- ``leiden_r{k}`` — the research strategies: every graph term replicated into
  its top-k affinity clusters (optionally thresholded by
  ``replica_affinity_quantile``).

Also stores the document frequency of every corpus term
(``data/interim/term_df.parquet``) — evaluate uses it to attribute the
duplication cost of replicas to frequent vs rare terms.
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from sharded_index import (
    ReplicatedTermPartition,
    TermPartition,
    replicate_with_volume_guard,
    tokenize,
)
from sharded_index.config import (
    PAIRS_PATH,
    PARTITIONS_DIR,
    TERM_DF_PATH,
    load_params,
    partition_config,
)
from sharded_index.dataset import docs_from_pairs, split_queries


def main() -> None:
    params = load_params()
    partition_params = params["partition"]
    replicas = partition_params.get("term_replicas", [])
    affinity_quantile = partition_params.get("replica_affinity_quantile")
    pairs = pd.read_parquet(PAIRS_PATH)

    # Граф — только по train-запросам (holdout остаётся для оценки),
    # с повторами, как они идут в парах.
    train_queries, holdout = split_queries(pairs, params["split"]["train_ratio"])
    graph_texts = pairs.loc[pairs["query"].isin(set(train_queries)), "query"].tolist()

    leiden_core = TermPartition.from_corpus(graph_texts, config=partition_config(params))

    # Токенизируем корпус один раз: документная частота (диагностика цены
    # репликации) + голоса со-встречаемости для affinity-fallback.
    doc_token_sets = [set(tokenize(text)) for text in docs_from_pairs(pairs).values()]
    term_df: Counter[str] = Counter()
    for tokens in doc_token_sets:
        term_df.update(tokens)
    corpus_terms = set(term_df)

    TERM_DF_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"term": list(term_df.keys()), "df": list(term_df.values())}
    ).to_parquet(TERM_DF_PATH, index=False)

    hash_partition = TermPartition.from_hashing(
        corpus_terms,
        n_shards=leiden_core.n_shards,
        node_strength=leiden_core.node_strength,
    )
    leiden_partition = leiden_core.with_hash_fallback(corpus_terms)

    leiden_core.save(PARTITIONS_DIR / "leiden_core")
    hash_partition.save(PARTITIONS_DIR / "hash")
    leiden_partition.save(PARTITIONS_DIR / "leiden")

    print(f"queries: {len(train_queries):,} train / {len(holdout):,} holdout")
    print(f"leiden_core: {leiden_core.n_shards} clusters, {leiden_core.n_terms:,} graph terms")
    print(f"corpus terms: {len(corpus_terms):,} "
          f"(fallback for {len(corpus_terms) - leiden_core.n_terms:,})")
    print(f"hash: {hash_partition.n_shards} shards | leiden: {leiden_partition.n_shards} shards")

    quantile_note = f", affinity_quantile={affinity_quantile}" if affinity_quantile else ""
    for k in replicas:
        replicated = ReplicatedTermPartition.from_partition(
            leiden_partition, k, affinity_quantile=affinity_quantile,
        )
        replicated.save(PARTITIONS_DIR / f"leiden_r{k}")
        print(f"leiden_r{k}: replication factor {replicated.replication_factor:.3f}{quantile_note}")

    if not partition_params.get("doc_affinity_fallback", False):
        return

    # Семейство leiden_aff: термы вне графа уходят не в дизъюнктное hash-
    # пространство, а в семантический кластер, с которым чаще всего
    # со-встречаются в документах (hash — только для термов-сирот).
    leiden_aff = leiden_core.with_affinity_fallback(doc_token_sets)
    leiden_aff.save(PARTITIONS_DIR / "leiden_aff")
    residual = sum(
        1 for s in leiden_aff.term_to_shard.values() if s >= leiden_core.n_shards
    )
    print(f"leiden_aff: residual hash terms {residual:,} of {len(corpus_terms):,} "
          f"({residual / len(corpus_terms):.2%})")

    for k in replicas:
        replicated = ReplicatedTermPartition.from_partition(
            leiden_aff, k, affinity_quantile=affinity_quantile,
        )
        replicated.save(PARTITIONS_DIR / f"leiden_aff_r{k}")
        print(f"leiden_aff_r{k}: replication factor "
              f"{replicated.replication_factor:.3f}{quantile_note}")

    cap = partition_params.get("volume_cap_ratio")
    if not cap:
        return

    # Семейство leiden_bal: расщепление кластеров, чей документный объём
    # превышает бюджет cap (после affinity-fallback, чтобы cap видел
    # фактические шарды); репликация — только в шарды в пределах бюджета.
    cfg = partition_config(params)
    term_df_plain = dict(term_df)
    leiden_bal = leiden_aff.with_volume_cap(
        doc_token_sets,
        term_df_plain,
        max_volume_ratio=cap,
        resolution=cfg.leiden_resolution,
        seed=cfg.leiden_seed,
        growth=2.0,
        max_rounds=5,
        fallback_offset=leiden_core.n_shards,
    )
    orphan_terms = {
        t for t, s in leiden_aff.term_to_shard.items() if s >= leiden_core.n_shards
    }
    bal_semantic = (
        min(leiden_bal.term_to_shard[t] for t in orphan_terms)
        if orphan_terms else len(set(leiden_bal.term_to_shard.values()))
    )
    leiden_bal.save(PARTITIONS_DIR / "leiden_bal")
    meta = json.dumps({"semantic_shards": bal_semantic}) + "\n"
    (PARTITIONS_DIR / "leiden_bal" / "meta.json").write_text(meta)
    print(f"leiden_bal: cap {cap:.0%}, {bal_semantic} semantic clusters")

    for k in replicas:
        replicated = replicate_with_volume_guard(
            leiden_bal, k, doc_token_sets, term_df_plain,
            max_volume_ratio=cap, affinity_quantile=affinity_quantile,
        )
        replicated.save(PARTITIONS_DIR / f"leiden_bal_r{k}")
        (PARTITIONS_DIR / f"leiden_bal_r{k}" / "meta.json").write_text(meta)
        print(f"leiden_bal_r{k}: replication factor "
              f"{replicated.replication_factor:.3f} (volume-guarded)")


if __name__ == "__main__":
    main()
