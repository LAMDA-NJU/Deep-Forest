import copy
import numpy as np
from numpy.testing import assert_allclose
import pytest

from deepforest._binner import Binner, _find_binning_thresholds_per_feature


kwargs = {
    "n_bins": 255,
    "bin_subsample": 2e5,
    "bin_type": "percentile",
    "random_state": 0,
}


def test_find_binning_thresholds_regular_data():
    data = np.linspace(0, 10, 1001)

    # Percentile
    bin_thresholds = _find_binning_thresholds_per_feature(data, n_bins=10)
    assert_allclose(bin_thresholds, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    bin_thresholds = _find_binning_thresholds_per_feature(data, n_bins=5)
    assert_allclose(bin_thresholds, [2, 4, 6, 8])

    # Interval
    bin_thresholds = _find_binning_thresholds_per_feature(
        data, n_bins=10, bin_type="interval"
    )
    assert_allclose(bin_thresholds, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    bin_thresholds = _find_binning_thresholds_per_feature(
        data, n_bins=5, bin_type="interval"
    )
    assert_allclose(bin_thresholds, [2, 4, 6, 8])


def test_find_binning_thresholds_invalid_binner_type():
    data = np.linspace(0, 10, 1001)

    err_msg = "Unknown binning type: unknown."
    with pytest.raises(ValueError, match=err_msg):
        _find_binning_thresholds_per_feature(
            data, n_bins=10, bin_type="unknown"
        )


def test_find_binning_thresholds_invalid_data_shape():
    data = np.zeros((10, 2))

    with pytest.raises(RuntimeError) as execinfo:
        _find_binning_thresholds_per_feature(data, n_bins=10)
    assert "Per-feature data should be of the shape" in str(execinfo.value)


@pytest.mark.parametrize(
    "param",
    [
        (0, {"n_bins": 1}),
        (1, {"bin_subsample": 0}),
        (2, {"bin_type": "unknown"}),
    ],
)
def test_binner_invalid_params(param):
    data = np.linspace(0, 10, 1001)
    case_kwargs = copy.deepcopy(kwargs)
    case_kwargs.update(param[1])

    binner = Binner(**case_kwargs)

    with pytest.raises(ValueError) as excinfo:
        binner.fit(data)

    if param[0] == 0:
        assert "should be in the range [2, 255]" in str(excinfo.value)
    elif param[0] == 1:
        assert "samples used to construct the Binner" in str(excinfo.value)
    elif param[0] == 2:
        assert "The type of binner should be one of" in str(excinfo.value)


def test_binner_transform_before_fitting():
    data = np.linspace(0, 10, 1001)
    binner = Binner(**kwargs)

    err_msg = "The binner has not been fitted yet when calling `transform`."
    with pytest.raises(RuntimeError, match=err_msg):
        binner.transform(data)
