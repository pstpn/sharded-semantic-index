"""Stage 1: MS MARCO → normalized query-document pairs (data/processed/pairs.parquet)."""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from sharded_index.config import PAIRS_PATH, load_params
from sharded_index.dataset import extract_query_doc_pairs


def main() -> None:
    params = load_params()["dataset"]

    dataset = load_dataset(params["hf_name"], params["hf_config"])
    pairs = extract_query_doc_pairs(
        dataset["train"],
        max_rows=params["max_rows"],
        max_passages_per_query=params["max_passages_per_query"],
        selected_only=params["selected_only"],
    )

    frame = pd.DataFrame(pairs)
    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(PAIRS_PATH, index=False)
    print(f"{len(frame):,} pairs, {frame['doc_id'].nunique():,} docs, "
          f"{frame['query'].nunique():,} queries -> {PAIRS_PATH}")


if __name__ == "__main__":
    main()
