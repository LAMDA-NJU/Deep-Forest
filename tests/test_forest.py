import pytest

from deepforest import RandomForestClassifier
from deepforest import ExtraTreesClassifier
from deepforest import RandomForestRegressor
from deepforest import ExtraTreesRegressor
from deepforest.forest import _get_n_samples_bootstrap

# Load utils
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.datasets import load_iris, load_wine, load_boston
from sklearn.ensemble._forest import (
    _get_n_samples_bootstrap as sklearn_get_n_samples_bootstrap,
)


@pytest.mark.parametrize("max_samples", [0.42, 42, None])
def test_n_samples_bootstrap(max_samples):
    n_samples = 420
    actual = _get_n_samples_bootstrap(n_samples, max_samples)
    expected = sklearn_get_n_samples_bootstrap(n_samples, max_samples)

    assert actual == expected


def test_n_samples_bootstrap_int_out_of_range():
    n_samples = 42
    max_samples = 420

    err_msg = "`max_samples` must be in range 1 to 42 but got value 420"
    with pytest.raises(ValueError, match=err_msg):
        _get_n_samples_bootstrap(n_samples, max_samples)


def test_n_samples_bootstrap_float_out_of_range():
    n_samples = 42
    max_samples = 1.1

    with pytest.raises(ValueError) as execinfo:
        _get_n_samples_bootstrap(n_samples, max_samples)
    assert "`max_samples` must be in range (0, 1)" in str(execinfo.value)


def test_n_samples_bootstrap_invalid_type():
    n_samples = 42
    max_samples = "42"

    err_msg = (
        "`max_samples` should be int or float, but got type" " '<class 'str'>'"
    )
    with pytest.raises(TypeError, match=err_msg):
        _get_n_samples_bootstrap(n_samples, max_samples)


@pytest.mark.parametrize("load_func", [load_iris, load_wine])
def test_forest_classifier_workflow(load_func):

    n_estimators = 100  # to avoid oob warning
    random_state = 42

    X, y = load_func(return_X_y=True)

    # Data binning
    binner = _BinMapper(random_state=random_state)
    X_binned = binner.fit_transform(X)

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )

    model.fit(X_binned, y)
    model.predict(X_binned)

    # Extremely Random Forest
    model = ExtraTreesClassifier(
        n_estimators=n_estimators, random_state=random_state
    )

    model.fit(X_binned, y)
    model.predict(X_binned)


@pytest.mark.parametrize("load_func", [load_boston])
def test_forest_regressor_workflow(load_func):

    n_estimators = 100  # to avoid oob warning
    random_state = 42

    X, y = load_func(return_X_y=True)

    # Data binning
    binner = _BinMapper(random_state=random_state)
    X_binned = binner.fit_transform(X)

    # Random Forest
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state
    )

    model.fit(X_binned, y)
    model.predict(X_binned)

    # Extremely Random Forest
    model = ExtraTreesRegressor(
        n_estimators=n_estimators, random_state=random_state
    )

    model.fit(X_binned, y)
    model.predict(X_binned)
