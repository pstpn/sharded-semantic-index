"""Stage 7: topical-coherence benchmark.

Average metrics hide WHEN semantic sharding wins.  This stage answers it
two ways, for every strategy:

- Real holdout queries are sliced by the connectivity of their own terms in
  the train graph — mean pairwise edge weight over the query's graph-term
  pairs.  Buckets: ``Q1``..``Q4`` (connectivity quartiles; every term is a
  graph term), ``single`` (fewer than two graph terms), ``fallback`` (the
  query contains out-of-graph terms).  Q4 imitates popular, well-supported
  queries; Q1 is near-OOD.  qrels exist for these, so recall@1 is measured.
- Synthetic probes with controlled connectivity (the causal check):
  ``connected_pairs`` (top intra-cluster edges), ``connected_triples``
  (edge plus the common neighbor maximizing the weaker link) and
  ``random_pairs`` (terms of different clusters, seeded rng).  No qrels —
  fanout, share@1 and probed volume only.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sharded_index import TermPartition, load_partition
from sharded_index.config import (
    ASSIGNMENTS_DIR,
    METRICS_DIR,
    PAIRS_PATH,
    PARTITIONS_DIR,
    load_params,
    strategy_names,
)
from sharded_index.dataset import docs_from_pairs, split_queries
from sharded_index.text import tokenize


def bucket_metrics(part, queries, qrels, doc_shards, shard_sizes, n_docs):
    """Routing metrics of one query bucket (qrels-recall when qrels given)."""
    fanouts, vol1, q1 = [], [], []
    for q in queries:
        cover = part.cover_shards_for_query(q)
        fanouts.append(len(cover))
        vol1.append(shard_sizes.get(cover[0], 0) / n_docs if cover else 0.0)
        gt = qrels.get(q)
        if gt:
            first = cover[0] if cover else None
            q1.append(
                sum(1 for d in gt if first is not None and first in doc_shards[d])
                / len(gt)
            )
    fanouts_arr = np.array(fanouts, dtype=float)
    entry = {
        "n": len(queries),
        "mean_fanout": round(float(fanouts_arr.mean()), 3) if len(queries) else None,
        "share_covered_by_1": round(float((fanouts_arr <= 1).mean()), 4)
        if len(queries) else None,
        "probed_volume_at1_mean": round(float(np.mean(vol1)), 4) if vol1 else None,
    }
    if q1:
        entry["qrels_recall_at_1"] = round(float(np.mean(q1)), 4)
    return entry


def connectivity_buckets(queries, graph_terms, pair_weight):
    """Split queries into fallback / single / Q1..Q4 by term connectivity."""
    scored, buckets = [], defaultdict(list)
    for q in queries:
        tokens = set(tokenize(q))
        graph_part = [t for t in tokens if t in graph_terms]
        if len(graph_part) < len(tokens):
            buckets["fallback"].append(q)
        elif len(graph_part) < 2:
            buckets["single"].append(q)
        else:
            weights = [
                pair_weight.get(frozenset((a, b)), 0.0)
                for i, a in enumerate(graph_part) for b in graph_part[i + 1:]
            ]
            scored.append((q, float(np.mean(weights))))
    bounds = np.quantile([s for _, s in scored], [0.25, 0.5, 0.75]) if scored else []
    for q, s in scored:
        tier = ("Q1" if s <= bounds[0] else "Q2" if s <= bounds[1]
                else "Q3" if s <= bounds[2] else "Q4")
        buckets[tier].append(q)
    return buckets, [round(float(b), 4) for b in bounds]


def synthetic_probes(core: TermPartition, n_probes: int, seed: int):
    """Pseudo-queries with controlled term connectivity."""
    edges = core.edges_df
    same = edges["src"].map(core.term_to_shard) == edges["dst"].map(core.term_to_shard)
    intra = edges[same].nlargest(n_probes, "weight")
    pairs = [f"{s} {d}" for s, d in zip(intra["src"], intra["dst"])]

    adj: dict[str, dict[str, float]] = defaultdict(dict)
    for s, d, w in zip(edges["src"], edges["dst"], edges["weight"]):
        adj[s][d] = float(w)
        adj[d][s] = float(w)
    triples = []
    for s, d in zip(intra["src"], intra["dst"]):
        common = set(adj[s]) & set(adj[d])
        if common:
            third = max(common, key=lambda t: (min(adj[s][t], adj[d][t]), t))
            triples.append(f"{s} {d} {third}")
    triples = triples[:n_probes]

    rng = np.random.default_rng(seed)
    terms = sorted(core.term_to_shard)
    random_pairs = []
    while len(random_pairs) < n_probes:
        a, b = rng.choice(len(terms), size=2, replace=False)
        ta, tb = terms[a], terms[b]
        if core.term_to_shard[ta] != core.term_to_shard[tb]:
            random_pairs.append(f"{ta} {tb}")
    return {"connected_pairs": pairs, "connected_triples": triples,
            "random_pairs": random_pairs}


def main() -> None:
    params = load_params()
    topical_params = params.get("analyze_topical", {})
    n_probes = topical_params.get("n_probes", 2000)
    probe_seed = topical_params.get("probe_seed", 42)

    pairs = pd.read_parquet(PAIRS_PATH)
    _, holdout = split_queries(pairs, params["split"]["train_ratio"])
    queries = holdout[: params["evaluate"]["n_eval"]]
    qrels_all = pairs.groupby("query")["doc_id"].agg(set).to_dict()
    qrels = {q: qrels_all[q] for q in queries}

    docs = docs_from_pairs(pairs)
    n_docs = len(docs)

    core = TermPartition.load(PARTITIONS_DIR / "leiden_core")
    graph_terms = set(core.term_to_shard)
    pair_weight = {
        frozenset((s, d)): float(w)
        for s, d, w in zip(core.edges_df["src"], core.edges_df["dst"],
                           core.edges_df["weight"])
    }

    buckets, bounds = connectivity_buckets(queries, graph_terms, pair_weight)
    probes = synthetic_probes(core, n_probes, probe_seed)
    print(f"buckets: { {k: len(v) for k, v in sorted(buckets.items())} } "
          f"| bounds {bounds}", flush=True)

    result: dict = {
        "split": "holdout",
        "bucket_sizes": {k: len(v) for k, v in sorted(buckets.items())},
        "connectivity_quantiles": bounds,
        "n_probes": {k: len(v) for k, v in probes.items()},
        "real": {},
        "synthetic": {},
    }
    for strategy in tqdm(strategy_names(params), desc="strategies"):
        part = load_partition(PARTITIONS_DIR / strategy)
        assignment = pd.read_parquet(ASSIGNMENTS_DIR / f"{strategy}.parquet")
        shard_sizes = assignment.groupby("shard")["doc_id"].nunique().to_dict()
        doc_shards: dict[str, set[int]] = defaultdict(set)
        for doc_id, shard in zip(assignment["doc_id"], assignment["shard"]):
            doc_shards[doc_id].add(int(shard))
        doc_shards = dict(doc_shards)

        result["real"][strategy] = {
            bucket: bucket_metrics(part, qlist, qrels, doc_shards, shard_sizes, n_docs)
            for bucket, qlist in sorted(buckets.items())
        }
        result["synthetic"][strategy] = {
            probe: bucket_metrics(part, plist, {}, doc_shards, shard_sizes, n_docs)
            for probe, plist in probes.items()
        }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "topical_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n",
    )
    print(json.dumps(
        {s: result["synthetic"][s]["connected_pairs"] for s in result["synthetic"]},
        indent=1,
    ), flush=True)


if __name__ == "__main__":
    main()
