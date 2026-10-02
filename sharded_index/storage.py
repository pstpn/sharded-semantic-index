"""Byte-stable array files.

``numpy.savez`` stamps archive members with the current time, so equal
arrays produce different files and DVC re-runs everything downstream.
These helpers write the same ``.npz`` format with a fixed timestamp.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_EPOCH = (1980, 1, 1, 0, 0, 0)


def save_arrays(path: Path, **arrays: np.ndarray) -> None:
    """Write named arrays to a compressed ``.npz`` readable by ``numpy.load``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            member = zipfile.ZipInfo(f"{name}.npy", date_time=_EPOCH)
            member.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(member, "w", force_zip64=True) as file:
                np.lib.format.write_array(file, np.asanyarray(array), allow_pickle=False)


def save_incidence(path: Path, matrix: sp.spmatrix) -> None:
    """Write a binary sparse matrix (only the positions of its non-zeros)."""
    csr = sp.csr_matrix(matrix)
    csr.sort_indices()
    save_arrays(
        path,
        indptr=csr.indptr.astype(np.int64),
        indices=csr.indices.astype(np.int32),
        shape=np.asarray(csr.shape, dtype=np.int64),
    )


def load_incidence(path: Path, dtype: type = bool) -> sp.csr_matrix:
    """Read a matrix written by :func:`save_incidence`."""
    with np.load(path) as data:
        indices = data["indices"]
        return sp.csr_matrix(
            (np.ones(len(indices), dtype=dtype), indices, data["indptr"]),
            shape=tuple(data["shape"]),
        )
