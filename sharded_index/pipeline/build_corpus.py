"""Tokenize the corpus once: the document-term incidence and the vocabulary."""

from __future__ import annotations

import logging

from sharded_index.config import Config
from sharded_index.data.corpus import Corpus
from sharded_index.paths import Paths
from sharded_index.pipeline.common import read_documents, setup

logger = logging.getLogger(__name__)


def run(config: Config, paths: Paths) -> None:
    corpus = Corpus.from_documents(read_documents(paths), config.tokenizer())
    corpus.save(paths.corpus)
    logger.info(
        "%d documents, %d terms, %d document-term pairs",
        corpus.n_docs,
        len(corpus.vocabulary),
        corpus.doc_terms.nnz,
    )


if __name__ == "__main__":
    run(*setup())
