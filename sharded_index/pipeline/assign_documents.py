"""Assign every document to the shards of its terms, for every strategy of one method."""

from __future__ import annotations

import logging

from sharded_index.config import Config
from sharded_index.data.corpus import Corpus
from sharded_index.evaluation.membership import assign_documents
from sharded_index.paths import Paths
from sharded_index.pipeline.common import method_argument, read_partitions, setup
from sharded_index.storage import save_incidence

logger = logging.getLogger(__name__)


def run(config: Config, paths: Paths, method: str) -> None:
    corpus = Corpus.load(paths.corpus)
    for name, partition in read_partitions(config, paths, method).items():
        membership = assign_documents(partition, corpus)
        save_incidence(paths.assignments(method) / f"{name}.npz", membership)
        logger.info("%s/%s: duplication %.2f", method, name, membership.nnz / corpus.n_docs)


if __name__ == "__main__":
    config, paths = setup()
    run(config, paths, method_argument(config))
