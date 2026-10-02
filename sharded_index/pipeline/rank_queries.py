"""Rank every sample query with the unsharded index.

The rankings directory persists between runs: a query ranked before with the
same index, depth and BM25 settings is not searched again.
"""

from __future__ import annotations

from sharded_index.config import Config
from sharded_index.data.corpus import Corpus
from sharded_index.index.build import FINGERPRINT_FILE
from sharded_index.index.ranking import ReferenceIndex, ranking_key, update_rankings
from sharded_index.paths import Paths
from sharded_index.pipeline.common import read_samples, setup


def run(config: Config, paths: Paths) -> None:
    fingerprint = (paths.index / FINGERPRINT_FILE).read_text().strip()
    key = ranking_key(fingerprint, config.evaluation.ranking_depth, config.bm25)
    reference = ReferenceIndex(paths.index, Corpus.load(paths.corpus).doc_ids, config.bm25)
    update_rankings(
        paths.rankings,
        read_samples(config, paths),
        reference,
        config.evaluation.ranking_depth,
        key,
    )
    reference.close()


if __name__ == "__main__":
    run(*setup())
