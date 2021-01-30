import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from deepforest import _utils


# Toy data
X = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]), dtype=np.float64)

X_uint8 = np.array(
    ([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]), dtype=np.uint8
)

X_aug = np.array(([13, 14], [15, 16], [17, 18], [19, 20]), dtype=np.float64)

X_aug_uint8 = np.array(
    ([13, 14], [15, 16], [17, 18], [19, 20]), dtype=np.uint8
)

X_proba = np.array(
    ([0.1, 0.9, 0.3, 0.7], [0.2, 0.8, 0.4, 0.6], [0.3, 0.7, 0.5, 0.5]),
    dtype=np.float64,
)


def test_merge_proba_normal():
    actual = _utils.merge_proba(X_proba, 2)
    expected = np.array(([0.2, 0.8], [0.3, 0.7], [0.4, 0.6]), dtype=np.float64)

    assert_allclose(actual, expected, rtol=1e-010)


def test_merge_proba_invalid_n_output():
    err_msg = "The dimension of probas = 4 does not match n_outputs = 3."
    with pytest.raises(RuntimeError, match=err_msg):
        _utils.merge_proba(X_proba, 3)


def test_init_array_normal():
    actual = _utils.init_array(X_uint8, 2)
    expected = np.array(
        (
            [1, 2, 3, 0, 0],
            [4, 5, 6, 0, 0],
            [7, 8, 9, 0, 0],
            [10, 11, 12, 0, 0],
        ),
        dtype=np.uint8,
    )

    assert_array_equal(actual, expected)


def test_init_array_invalid_dtype():
    err_msg = "The input `X` when creating the array should be binned."
    with pytest.raises(ValueError, match=err_msg):
        _utils.init_array(X, 2)


def test_merge_array_normal():
    n_features = 3
    n_aug_features = 2

    X_middle = _utils.init_array(X_uint8, n_aug_features)
    actual = _utils.merge_array(X_middle, X_aug_uint8, n_features)

    expected = np.array(
        (
            [1, 2, 3, 13, 14],
            [4, 5, 6, 15, 16],
            [7, 8, 9, 17, 18],
            [10, 11, 12, 19, 20],
        ),
        dtype=np.uint8,
    )

    assert_array_equal(actual, expected)


def test_merge_array_invalid_dtype():
    n_features = 3
    err_msg = "The input `X_aug` when merging the array should be binned."
    with pytest.raises(ValueError, match=err_msg):
        _utils.merge_array(X_uint8, X_aug, n_features)
