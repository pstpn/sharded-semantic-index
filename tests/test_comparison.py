"""Complete comparisons must retain every strategy and align by keys."""

import numpy as np
import pandas as pd
import pytest

from sharded_index.reporting.comparison import matrix


def test_matrix_aligns_all_strategies():
    frame = pd.DataFrame(
        [
            ("m", "aff", "b", 4),
            ("m", "base", "a", 1),
            ("m", "aff", "a", 3),
            ("m", "base", "b", 2),
        ],
        columns=["method", "strategy", "sample", "value"],
    )
    result = matrix(frame, ["m"], ["base", "aff"], "sample", ["a", "b"], "value")
    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])


def test_matrix_rejects_missing_configuration():
    frame = pd.DataFrame([("m", "base", "a", 1)], columns=["method", "strategy", "sample", "value"])
    with pytest.raises(ValueError, match="Missing or non-finite"):
        matrix(frame, ["m"], ["base", "aff"], "sample", ["a"], "value")


def test_matrix_rejects_duplicate_measurement():
    frame = pd.DataFrame(
        [("m", "base", "a", 1)] * 2, columns=["method", "strategy", "sample", "value"]
    )
    with pytest.raises(ValueError, match="non-unique"):
        matrix(frame, ["m"], ["base"], "sample", ["a"], "value")
