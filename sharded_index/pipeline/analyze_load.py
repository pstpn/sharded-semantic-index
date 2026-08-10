"""Stage 5: shard-load and probed-volume analysis, uncertainty, seed sensitivity.

Fanout counts shards, not work: with heavy-tailed shard sizes one probe may
scan most of the corpus.  This stage converts routing decisions into volume
terms and quantifies what the fanout metric hides:

- probed volume: share of the corpus in the probed shards, at budget 1 and
  at the full greedy cover (sum over cover shards, in corpus equivalents);
- shard-size and probe-traffic concentration (top-1 / top-10 shares);
- a partition-independent lexical ceiling for qrels recall (share of qrels
  documents having at least one term in common with the query) — unlike
  full-routing qrels recall it contains no shard-collision component;
- bootstrap 95% CIs for the headline holdout metrics and paired deltas
  between adjacent strategies;
- Leiden seed sensitivity: clusters, duplication, fanout and qrels-recall@1
  re-derived from re-clustered partitions (no index building involved).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sharded_index import ReplicatedTermPartition, TermPartition, load_partition, tokenize
from sharded_index.clustering import cluster_terms
from sharded_index.config import (
    ASSIGNMENTS_DIR,
    METRICS_DIR,
    PAIRS_PATH,
    PARTITIONS_DIR,
    load_params,
    partition_config,
    strategy_names,
)
from sharded_index.dataset import docs_from_pairs, split_queries

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42


def _ci95(values: np.ndarray, idx: np.ndarray) -> list[float]:
    """Bootstrap 95% CI for the mean of ``values`` (idx: resample matrix)."""
    means = values[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4)]


def _qrels_recall_at_1(
    cover: list[int],
    gt_ids: set[str],
    doc_shards: dict[str, set[int]],
) -> float:
    """Share of qrels docs living in the first cover shard (empty cover → 0)."""
    if not gt_ids:
        return 1.0
    if not cover:
        return 0.0
    first = cover[0]
    return sum(1 for d in gt_ids if first in doc_shards[d]) / len(gt_ids)


def _top_share(counter: Counter, top: int) -> float:
    total = sum(counter.values())
    return round(sum(c for _, c in counter.most_common(top)) / max(total, 1), 4)


def _analyze_partition(
    partition,
    queries: list[str],
    qrels: dict[str, set[str]],
    doc_shards: dict[str, set[int]],
    shard_sizes: dict[int, int],
    n_docs: int,
    idx: np.ndarray | None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Routing of ``queries`` in volume terms + per-query arrays for CIs."""
    covers = [
        partition.cover_shards_for_query(q)
        for q in tqdm(queries, desc="covers", leave=False)
    ]

    fanout = np.array([len(c) for c in covers], dtype=np.float64)
    vol1 = np.array(
        [shard_sizes.get(c[0], 0) / n_docs if c else 0.0 for c in covers], dtype=np.float64,
    )
    vol_full = np.array(
        [sum(shard_sizes.get(s, 0) for s in c) / n_docs for c in covers], dtype=np.float64,
    )
    q1 = np.array(
        [_qrels_recall_at_1(c, qrels[q], doc_shards) for c, q in zip(covers, queries)],
        dtype=np.float64,
    )
    share1 = (fanout <= 1).astype(np.float64)

    probes_at1 = Counter(c[0] for c in covers if c)
    probes_full: Counter = Counter()
    for c in covers:
        probes_full.update(c)

    # Доли корпуса: топ-1 шард — доля документов; топ-10 — корпус-эквиваленты
    # (шарды пересекаются, сумма долей не ограничена единицей).
    sizes_desc = sorted(shard_sizes.values(), reverse=True)
    entry = {
        "mean_fanout": round(float(fanout.mean()), 4),
        "share_covered_by_1": round(float(share1.mean()), 4),
        "qrels_recall_at_1": round(float(q1.mean()), 4),
        "probed_volume_at1_mean": round(float(vol1.mean()), 4),
        "probed_volume_at1_median": round(float(np.median(vol1)), 4),
        "probed_volume_full_mean": round(float(vol_full.mean()), 4),
        "top1_doc_share": round(sizes_desc[0] / n_docs, 4) if sizes_desc else 0.0,
        "top10_doc_equiv": round(sum(sizes_desc[:10]) / n_docs, 4),
        "probe_share_top1_at1": _top_share(probes_at1, 1),
        "probe_share_top10_at1": _top_share(probes_at1, 10),
        "probe_share_top1_full": _top_share(probes_full, 1),
        "probe_share_top10_full": _top_share(probes_full, 10),
    }
    if idx is not None:
        entry["ci95"] = {
            "mean_fanout": _ci95(fanout, idx),
            "share_covered_by_1": _ci95(share1, idx),
            "qrels_recall_at_1": _ci95(q1, idx),
            "probed_volume_at1_mean": _ci95(vol1, idx),
        }
    arrays = {"fanout": fanout, "share1": share1, "q1": q1, "vol1": vol1}
    return entry, arrays


def _paired_delta(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], idx: np.ndarray,
) -> dict[str, dict]:
    """Paired bootstrap for per-query deltas (a − b) of the headline metrics."""
    out = {}
    for key in ("fanout", "share1", "q1", "vol1"):
        d = a[key] - b[key]
        out[key] = {"mean_delta": round(float(d.mean()), 4), "ci95": _ci95(d, idx)}
    return out


def _derived_metrics(
    part,
    doc_tokens: dict[str, set[str]],
    queries: list[str],
    qrels: dict[str, set[str]],
    n_docs: int,
) -> dict:
    """Metrics of a re-built partition, derived purely from tokens.

    Assignments and shard sizes come from ``shards_for_tokens`` over the
    tokenized corpus — physical indices are never built, so seed and
    resolution variants stay cheap.
    """
    doc_shards = {d: part.shards_for_tokens(t) for d, t in doc_tokens.items()}
    shard_sizes: Counter = Counter()
    for shards in doc_shards.values():
        shard_sizes.update(shards)
    dup = float(np.mean([len(s) for s in doc_shards.values()]))
    entry, _ = _analyze_partition(
        part, queries, qrels, doc_shards, dict(shard_sizes), n_docs, None,
    )
    entry["duplication"] = round(dup, 2)
    return entry


def main() -> None:
    params = load_params()
    eval_params = params["evaluate"]
    analyze_params = params.get("analyze_load", {})
    seeds = analyze_params.get("leiden_seeds", [1, 2, 3])
    n_bootstrap = analyze_params.get("bootstrap_samples", BOOTSTRAP_SAMPLES)

    pairs = pd.read_parquet(PAIRS_PATH)
    _, holdout_queries = split_queries(pairs, params["split"]["train_ratio"])
    queries = holdout_queries[: eval_params["n_eval"]]
    qrels_all = pairs.groupby("query")["doc_id"].agg(set).to_dict()
    qrels = {q: qrels_all[q] for q in queries}

    docs = docs_from_pairs(pairs)
    n_docs = len(docs)
    print(f"tokenizing corpus: {n_docs:,} docs")
    doc_tokens = {doc_id: set(tokenize(text)) for doc_id, text in tqdm(docs.items(), leave=False)}
    corpus_terms: set[str] = set()
    for tokens in doc_tokens.values():
        corpus_terms.update(tokens)

    # ── Партиционно-независимый лексический потолок qrels ─────────────
    ceiling_vals = []
    for q in queries:
        qt = set(tokenize(q))
        gt = qrels[q]
        ceiling_vals.append(sum(1 for d in gt if doc_tokens[d] & qt) / len(gt))
    lexical_ceiling = round(float(np.mean(ceiling_vals)), 4)
    print(f"lexical ceiling (holdout qrels): {lexical_ceiling}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(queries), size=(n_bootstrap, len(queries)), dtype=np.int32)

    strategies = strategy_names(params)
    result: dict = {
        "split": "holdout",
        "n_queries": len(queries),
        "n_docs": n_docs,
        "bootstrap_samples": n_bootstrap,
        "lexical_ceiling_qrels": lexical_ceiling,
        "strategies": {},
    }
    arrays_by_strategy: dict[str, dict[str, np.ndarray]] = {}

    for strategy in strategies:
        print(f"[{strategy}] routing {len(queries):,} holdout queries")
        partition = load_partition(PARTITIONS_DIR / strategy)
        assignment = pd.read_parquet(ASSIGNMENTS_DIR / f"{strategy}.parquet")
        shard_sizes = assignment.groupby("shard")["doc_id"].nunique().to_dict()
        doc_shards: dict[str, set[int]] = defaultdict(set)
        for doc_id, shard in zip(assignment["doc_id"], assignment["shard"]):
            doc_shards[doc_id].add(int(shard))

        entry, arrays = _analyze_partition(
            partition, queries, qrels, dict(doc_shards), shard_sizes, n_docs, idx,
        )
        entry["duplication"] = round(len(assignment) / assignment["doc_id"].nunique(), 2)
        result["strategies"][strategy] = entry
        arrays_by_strategy[strategy] = arrays

    result["paired_deltas"] = {
        "leiden_aff_r3_minus_leiden_r3": _paired_delta(
            arrays_by_strategy["leiden_aff_r3"], arrays_by_strategy["leiden_r3"], idx,
        ),
        "leiden_aff_r3_minus_hash": _paired_delta(
            arrays_by_strategy["leiden_aff_r3"], arrays_by_strategy["hash"], idx,
        ),
    }

    # ── Варианты кластеризации: seed и разрешение γ ───────────────────
    # Разбиения пересобираются из сохранённого графа (edges leiden_core);
    # индексы не строятся: назначения и размеры шардов выводятся из токенов.
    gammas = analyze_params.get("gamma_sweep", [2, 5, 10, 20, 50])
    _SEED_KEYS = (
        "duplication", "mean_fanout", "share_covered_by_1",
        "qrels_recall_at_1", "top1_doc_share",
    )
    if seeds or gammas:
        core = TermPartition.load(PARTITIONS_DIR / "leiden_core")
        cfg = partition_config(params)
        doc_token_list = list(doc_tokens.values())

    if seeds:
        seed_block: dict = {
            "seeds": [cfg.leiden_seed, *seeds],
            "n_clusters": [core.n_shards],
            "strategies": {},
        }
        variants: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for seed in seeds:
            print(f"[seed {seed}] re-clustering graph")
            term_to_shard, clusters_df = cluster_terms(
                core.edges_df, resolution=cfg.leiden_resolution, seed=seed,
            )
            core_s = TermPartition(term_to_shard, core.node_strength, core.edges_df, clusters_df)
            seed_block["n_clusters"].append(core_s.n_shards)
            aff_s = core_s.with_affinity_fallback(doc_token_list)
            parts = {
                "leiden": core_s.with_hash_fallback(corpus_terms),
                "leiden_aff_r3": ReplicatedTermPartition.from_partition(aff_s, 3),
            }
            for name, part in parts.items():
                entry = _derived_metrics(part, doc_tokens, queries, qrels, n_docs)
                for key in _SEED_KEYS:
                    variants[name][key].append(entry[key])

        # Первый элемент каждого списка — базовый seed (артефакты конвейера).
        for name in variants:
            base = result["strategies"][name]
            seed_block["strategies"][name] = {
                key: [base[key], *variants[name][key]] for key in _SEED_KEYS
            }
        result["seed_sensitivity"] = seed_block

    # γ-свип: управляемость баланса размеров шардов разрешением кластеризации.
    # Строится только конфигурация-фаворит (leiden_aff ×3).
    if gammas:
        base = result["strategies"]["leiden_aff_r3"]
        base_row = {"gamma": cfg.leiden_resolution, "n_clusters": core.n_shards}
        base_row.update({key: base[key] for key in _SEED_KEYS})
        base_row["probed_volume_at1_mean"] = base["probed_volume_at1_mean"]
        rows = [base_row]
        for gamma in gammas:
            print(f"[gamma {gamma}] re-clustering graph")
            term_to_shard, clusters_df = cluster_terms(
                core.edges_df, resolution=float(gamma), seed=cfg.leiden_seed,
            )
            core_g = TermPartition(term_to_shard, core.node_strength, core.edges_df, clusters_df)
            aff_g = core_g.with_affinity_fallback(doc_token_list)
            part_g = ReplicatedTermPartition.from_partition(aff_g, 3)
            entry = _derived_metrics(part_g, doc_tokens, queries, qrels, n_docs)
            row = {"gamma": gamma, "n_clusters": core_g.n_shards}
            row.update({key: entry[key] for key in _SEED_KEYS})
            row["probed_volume_at1_mean"] = entry["probed_volume_at1_mean"]
            rows.append(row)
        result["gamma_sweep"] = {"strategy": "leiden_aff_r3", "rows": rows}

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "load_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({s: result["strategies"][s] for s in strategies}, indent=2))


if __name__ == "__main__":
    main()
