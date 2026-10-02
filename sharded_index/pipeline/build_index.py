"""Build the unsharded BM25 index, the reference of the evaluation.

The index directory persists between runs and is rebuilt only when the
corpus or the stop list changes (see ``corpus_fingerprint``).
"""

from __future__ import annotations

import logging

from sharded_index.config import Config
from sharded_index.index.build import FINGERPRINT_FILE, build_index, corpus_fingerprint
from sharded_index.paths import Paths
from sharded_index.pipeline.common import read_documents, setup

logger = logging.getLogger(__name__)


def run(config: Config, paths: Paths) -> None:
    documents = read_documents(paths)
    fingerprint = corpus_fingerprint(documents, config.text.stop_words)
    marker = paths.index / FINGERPRINT_FILE
    if marker.exists() and marker.read_text().strip() == fingerprint:
        logger.info("the index matches the corpus, nothing to build")
        return
    build_index(documents, paths.index, config.text.stop_words)
    marker.write_text(fingerprint + "\n")
    logger.info("indexed %d documents", len(documents))


if __name__ == "__main__":
    run(*setup())
