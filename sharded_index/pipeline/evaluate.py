"""Stage 4: evaluation on three query populations (splits).

- ``train``   — queries the NPMI graph was built on: the partition's capacity,
  an upper bound (even here cover fanout is not 1 — cross-topic queries).
- ``holdout`` — unseen queries of the same dataset: in-distribution
  generalization, the **primary** tier (all main figures use it).
- ``ood``     — queries from a foreign dataset (``evaluate.ood`` in params):
  transfer to an alien distribution.  No qrels exist for them; fanout and
  BM25-recall still apply, and ``empty_gt_share`` shows how many queries
  match nothing at all.

Per split and strategy: cover fanout (incl. ``share_covered_by_1``), recall
at budgets over two reference sets (BM25 ground truth of the full index and,
where available, qrels), fallback/mixed-query shares.  Full-cover recall
against BM25 ground truth is 1.0 by construction — a sanity check.
"""

from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from sharded_index import TermPartition, index_size_mb, load_partition
from sharded_index.config import (
    ASSIGNMENTS_DIR,
    EVAL_DIR,
    INDICES_DIR,
    METRICS_DIR,
    PAIRS_PATH,
    PARTITIONS_DIR,
    TERM_DF_PATH,
    load_params,
    strategy_names,
)
from sharded_index.dataset import docs_from_pairs, split_queries
from sharded_index.evaluation import (
    build_ground_truth,
    build_ground_truth_fast,
    compute_routing_recall,
    cover_fanouts,
    recall_by_fanout,
    term_postings,
)
from sharded_index.text import normalize_text, tokenize


def fanout_stats(fanouts: list[int]) -> dict[str, float]:
    series = pd.Series(fanouts)
    return {
        "mean": round(float(series.mean()), 2),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        # Доля запросов, которым хватает одного шарда, — прямой ответ на
        # вопрос «достигли ли fanout = 1».
        "share_covered_by_1": round(float((series <= 1).mean()), 4),
    }


def fallback_term_share(partition, semantic_shards: int) -> float:
    """Share of vocabulary living in the hash-fallback space."""
    term_shards = getattr(partition, "term_to_shards", None)
    if term_shards is None:
        term_shards = {t: (s,) for t, s in partition.term_to_shard.items()}
    fallback = sum(1 for shards in term_shards.values() if shards[0] >= semantic_shards)
    return round(fallback / max(len(term_shards), 1), 4)


def routing_space_shares(partition, queries: list[str], semantic_shards: int) -> dict[str, float]:
    """Per-split routing diagnostics.

    ``mixed_query_share`` — queries touching both the semantic and the
    fallback space (disjoint ⇒ such a query can never be covered by one
    shard); the affinity fallback exists to shrink this.
    """
    probes = fallback_probes = mixed_queries = 0
    for query in queries:
        shards = partition.shards_for_query(query)
        probes += len(shards)
        fallback_probes += sum(1 for s in shards if s >= semantic_shards)
        if any(s < semantic_shards for s in shards) and any(s >= semantic_shards for s in shards):
            mixed_queries += 1
    return {
        "fallback_probe_share": round(fallback_probes / max(probes, 1), 4),
        "mixed_query_share": round(mixed_queries / max(len(queries), 1), 4),
    }


def replication_diagnostics(partition, term_df: dict[str, int]) -> dict:
    """Who pays for replication: df-weighted cost of extra replicas.

    The duplication a replica causes is proportional to the term's document
    frequency, so the cost of term t is ``df(t) * (n_shards(t) - 1)``; the
    shares show how much of it sits in the most frequent terms.
    """
    extras = {
        t: len(shards) - 1
        for t, shards in partition.term_to_shards.items()
        if len(shards) > 1
    }
    if not extras:
        return {"replicated_terms": 0}

    cost = {t: term_df.get(t, 0) * n for t, n in extras.items()}
    total_cost = sum(cost.values())
    df_sorted = sorted(term_df.values(), reverse=True)

    def top_share(pct: float) -> float:
        threshold = df_sorted[max(int(len(df_sorted) * pct) - 1, 0)]
        top = sum(c for t, c in cost.items() if term_df.get(t, 0) >= threshold)
        return round(top / max(total_cost, 1), 4)

    return {
        "replicated_terms": len(extras),
        "df_weighted_cost_share_top1pct": top_share(0.01),
        "df_weighted_cost_share_top10pct": top_share(0.10),
    }


def load_ood_queries(ood_params: dict) -> list[str]:
    """Foreign queries for the OOD tier, normalized and deduplicated."""
    from datasets import load_dataset

    dataset = load_dataset(
        ood_params["hf_name"], ood_params.get("hf_config"), split=ood_params["split"],
    )
    normalized = (normalize_text(t) for t in dataset[ood_params["text_field"]])
    return [q for q in dict.fromkeys(normalized) if q]


def main() -> None:
    params = load_params()
    eval_params = params["evaluate"]
    strategies = strategy_names(params)
    max_fanout = eval_params["max_fanout_sweep"]
    pairs = pd.read_parquet(PAIRS_PATH)

    # ── Три яруса оценки ──────────────────────────────────────────────
    train_queries, holdout_queries = split_queries(pairs, params["split"]["train_ratio"])
    populations: dict[str, list[str]] = {
        "train": train_queries[: eval_params["n_eval_train"]],
        "holdout": holdout_queries[: eval_params["n_eval"]],
    }
    ood_params = eval_params.get("ood")
    if ood_params:
        populations["ood"] = load_ood_queries(ood_params)[: eval_params["n_eval_ood"]]

    qrels_all = pairs.groupby("query")["doc_id"].agg(set).to_dict()
    has_qrels = {"train": True, "holdout": True, "ood": False}

    # fast_gt: эталон в памяти (документы с >=1 общим термом) вместо
    # небюджетного BM25 по физическому индексу. Эквивалентность закреплена
    # smoke-тестом (паритет tokenize и анализатора Whoosh); режим для быстрых
    # итераций разработки, публикационные числа — с fast_gt: false.
    fast_gt = bool(eval_params.get("fast_gt", False))
    ground_truth: dict[str, dict[str, set[str]]] = {}
    if fast_gt:
        docs = docs_from_pairs(pairs)
        print(f"[fast-gt] tokenizing {len(docs):,} docs")
        postings = term_postings({d: set(tokenize(t)) for d, t in docs.items()})
        for split, queries in populations.items():
            print(f"[{split}] ground truth (fast) for {len(queries):,} queries")
            ground_truth[split] = build_ground_truth_fast(queries, postings)
    else:
        for split, queries in populations.items():
            print(f"[{split}] ground truth for {len(queries):,} queries")
            ground_truth[split] = build_ground_truth(
                queries, INDICES_DIR / "full" / "shard_000",
            )

    term_df_frame = pd.read_parquet(TERM_DF_PATH)
    term_df = dict(zip(term_df_frame["term"].tolist(), term_df_frame["df"].tolist()))

    full_mb = index_size_mb(INDICES_DIR / "full")
    semantic_shards = TermPartition.load(PARTITIONS_DIR / "leiden_core").n_shards

    def semantic_boundary(strategy: str) -> int:
        """Граница семантического id-пространства стратегии.

        Стратегии с перенумерацией (leiden_bal*) хранят её в meta.json;
        остальные наследуют число кластеров leiden_core.
        """
        meta_path = PARTITIONS_DIR / strategy / "meta.json"
        if meta_path.exists():
            return int(json.loads(meta_path.read_text())["semantic_shards"])
        return semantic_shards

    metrics: dict = {
        "eval_splits": {split: len(queries) for split, queries in populations.items()},
        "gt_mode": "token_membership" if fast_gt else "bm25",
        "semantic_shards": semantic_shards,
        "full_index_mb": round(full_mb, 1),
    }
    curves: dict[str, dict[str, dict[int, float]]] = {split: {} for split in populations}
    fanout_frames = []

    for strategy in strategies:
        partition = load_partition(PARTITIONS_DIR / strategy)

        assignment = pd.read_parquet(ASSIGNMENTS_DIR / f"{strategy}.parquet")
        doc_shards: dict[str, set[int]] = defaultdict(set)
        for doc_id, shard in zip(assignment["doc_id"], assignment["shard"]):
            doc_shards[doc_id].add(int(shard))
        doc_shards = dict(doc_shards)

        size_mb = index_size_mb(INDICES_DIR / strategy)
        metrics[strategy] = {
            "n_shards": partition.n_shards,
            "shards_used": int(assignment["shard"].nunique()),
            "replication_factor": round(getattr(partition, "replication_factor", 1.0), 2),
            "duplication": round(len(assignment) / assignment["doc_id"].nunique(), 2),
            "index_mb": round(size_mb, 1),
            "storage_overhead": round(size_mb / full_mb, 2),
        }
        boundary = semantic_boundary(strategy)
        if strategy != "hash":
            metrics[strategy]["fallback_term_share"] = fallback_term_share(
                partition, boundary,
            )
        if hasattr(partition, "term_to_shards"):
            metrics[strategy]["replication"] = replication_diagnostics(partition, term_df)

        by_split: dict[str, dict] = {}
        for split, queries in populations.items():
            gt = ground_truth[split]
            routing = compute_routing_recall(queries, gt, partition, doc_shards)
            curve = recall_by_fanout(queries, gt, partition, doc_shards, max_fanout=max_fanout)
            curves[split][strategy] = curve
            cover = cover_fanouts(queries, partition)
            fanout_frames.append(
                pd.DataFrame({"strategy": strategy, "split": split, "fanout": cover})
            )

            entry: dict = {
                "n_queries": len(queries),
                "empty_gt_share": round(
                    sum(1 for q in queries if not gt.get(q)) / max(len(queries), 1), 4,
                ),
                "cover_fanout": fanout_stats(cover),
                "union_fanout_mean": round(routing.mean_fanout, 2),
                "routing_recall_full": round(routing.recall, 4),  # sanity: must be 1.0
                "recall_at": {str(f): round(r, 4) for f, r in curve.items()},
            }
            if has_qrels.get(split):
                qrels = {q: qrels_all[q] for q in queries}
                qrels_full = compute_routing_recall(queries, qrels, partition, doc_shards)
                qrels_curve = recall_by_fanout(
                    queries, qrels, partition, doc_shards, max_fanout=max_fanout,
                )
                # qrels-recall при полной маршрутизации ≈ лексический потолок
                # (чуть выше него за счёт случайных коллизий шардов).
                entry["qrels_recall_full"] = round(qrels_full.recall, 4)
                entry["qrels_recall_at"] = {str(f): round(r, 4) for f, r in qrels_curve.items()}
            if strategy != "hash":
                entry.update(routing_space_shares(partition, queries, boundary))
            by_split[split] = entry

        metrics[strategy]["by_split"] = by_split

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(fanout_frames, ignore_index=True).to_parquet(
        EVAL_DIR / "fanouts.parquet", index=False,
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (METRICS_DIR / "recall_by_fanout.json").write_text(json.dumps(
        {
            split: {s: {str(f): r for f, r in curve.items()} for s, curve in by_strategy.items()}
            for split, by_strategy in curves.items()
        },
        indent=2,
    ) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
