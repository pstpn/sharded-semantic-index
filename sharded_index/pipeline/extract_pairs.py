"""MS MARCO → normalized query-passage pairs."""

from __future__ import annotations

import logging

from datasets import load_dataset

from sharded_index.config import Config
from sharded_index.data.msmarco import extract_pairs
from sharded_index.paths import Paths
from sharded_index.pipeline.common import setup

logger = logging.getLogger(__name__)


def run(config: Config, paths: Paths) -> None:
    dataset = load_dataset(config.dataset.name, config.dataset.config)
    pairs = extract_pairs(
        dataset[config.dataset.split],
        max_rows=config.dataset.max_rows,
        docs_per_query=config.dataset.docs_per_query,
    )
    paths.pairs.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(paths.pairs, index=False)
    logger.info(
        "%d pairs, %d documents, %d queries",
        len(pairs),
        pairs["doc_id"].nunique(),
        pairs["query"].nunique(),
    )


if __name__ == "__main__":
    run(*setup())
