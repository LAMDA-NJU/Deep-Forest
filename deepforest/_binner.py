"""
Implementation of the Binner class in Deep Forest.

This class is modified from:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/_hist_gradient_boosting/binning.py
"""


__all__ = ["Binner"]

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array

from . import _cutils as _LIB


X_DTYPE = np.float64
X_BINNED_DTYPE = np.uint8
ALMOST_INF = 1e300


def _find_binning_thresholds_per_feature(
    col_data, n_bins, bin_type="percentile"
):
    """
    Private function used to find midpoints for samples along a
    specific feature.
    """
    if len(col_data.shape) != 1:

        msg = (
            "Per-feature data should be of the shape (n_samples,), but"
            " got {}-dims instead."
        )
        raise RuntimeError(msg.format(len(col_data.shape)))

    missing_mask = np.isnan(col_data)
    if missing_mask.any():
        col_data = col_data[~missing_mask]
    col_data = np.ascontiguousarray(col_data, dtype=X_DTYPE)
    distinct_values = np.unique(col_data)
    # Too few distinct values
    if len(distinct_values) <= n_bins:
        midpoints = distinct_values[:-1] + distinct_values[1:]
        midpoints *= 0.5
    else:
        # Equal interval in terms of percentile
        if bin_type == "percentile":
            percentiles = np.linspace(0, 100, num=n_bins + 1)
            percentiles = percentiles[1:-1]
            midpoints = np.percentile(
                col_data, percentiles, interpolation="midpoint"
            ).astype(X_DTYPE)
            assert midpoints.shape[0] == n_bins - 1
            np.clip(midpoints, a_min=None, a_max=ALMOST_INF, out=midpoints)
        # Equal interval in terms of value
        elif bin_type == "interval":
            min_value, max_value = np.min(col_data), np.max(col_data)
            intervals = np.linspace(min_value, max_value, num=n_bins + 1)
            midpoints = intervals[1:-1]
            assert midpoints.shape[0] == n_bins - 1
        else:
            raise ValueError("Unknown binning type: {}.".format(bin_type))

    return midpoints


def _find_binning_thresholds(
    X, n_bins, bin_subsample=2e5, bin_type="percentile", random_state=None
):
    n_samples, n_features = X.shape
    rng = check_random_state(random_state)

    if n_samples > bin_subsample:
        subset = rng.choice(np.arange(n_samples), bin_subsample, replace=False)
        X = X.take(subset, axis=0)

    binning_thresholds = []
    for f_idx in range(n_features):
        threshold = _find_binning_thresholds_per_feature(
            X[:, f_idx], n_bins, bin_type
        )
        binning_thresholds.append(threshold)

    return binning_thresholds


class Binner(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_bins=255,
        bin_subsample=2e5,
        bin_type="percentile",
        random_state=None,
    ):
        self.n_bins = n_bins + 1  # + 1 for missing values
        self.bin_subsample = int(bin_subsample)
        self.bin_type = bin_type
        self.random_state = random_state
        self._is_fitted = False

    def _validate_params(self):

        if not 2 <= self.n_bins - 1 <= 255:
            msg = (
                "`n_bins` should be in the range [2, 255], bug got"
                " {} instead."
            )
            raise ValueError(msg.format(self.n_bins - 1))

        if not self.bin_subsample > 0:
            msg = (
                "The number of samples used to construct the Binner"
                " should be strictly positive, but got {} instead."
            )
            raise ValueError(msg.format(self.bin_subsample))

        if self.bin_type not in ("percentile", "interval"):
            msg = (
                "The type of binner should be one of {{percentile, interval"
                "}}, bug got {} instead."
            )
            raise ValueError(msg.format(self.bin_type))

    def fit(self, X):

        self._validate_params()

        self.bin_thresholds_ = _find_binning_thresholds(
            X,
            self.n_bins - 1,
            self.bin_subsample,
            self.bin_type,
            self.random_state,
        )

        self.n_bins_non_missing_ = np.array(
            [thresholds.shape[0] + 1 for thresholds in self.bin_thresholds_],
            dtype=np.uint32,
        )

        self.missing_values_bin_idx_ = self.n_bins - 1
        self._is_fitted = True

        return self

    def transform(self, X):

        if not self._is_fitted:
            msg = (
                "The binner has not been fitted yet when calling `transform`."
            )
            raise RuntimeError(msg)

        if not X.shape[1] == self.n_bins_non_missing_.shape[0]:
            msg = (
                "The binner was fitted with {} features but {} features got"
                " passed to `transform`."
            )
            raise ValueError(
                msg.format(self.n_bins_non_missing_.shape[0], X.shape[1])
            )

        X = check_array(X, dtype=X_DTYPE, force_all_finite=False)
        X_binned = np.zeros_like(X, dtype=X_BINNED_DTYPE, order="F")

        _LIB._map_to_bins(
            X, self.bin_thresholds_, self.missing_values_bin_idx_, X_binned
        )

        return X_binned
