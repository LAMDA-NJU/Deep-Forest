import copy
import pytest
import shutil
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import deepforest
from deepforest import CascadeForestRegressor
from deepforest.cascade import _get_predictor_kwargs


save_dir = "./tmp"

# Load data
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.42, random_state=42
)

# Parameters
toy_kwargs = {
    "n_bins": 10,
    "bin_subsample": 2e5,
    "max_layers": 10,
    "criterion": "mse",
    "n_estimators": 1,
    "n_trees": 100,
    "max_depth": 3,
    "min_samples_leaf": 1,
    "use_predictor": True,
    "predictor": "forest",
    "predictor_kwargs": {},
    "n_tolerant_rounds": 2,
    "delta": 1e-5,
    "n_jobs": -1,
    "random_state": 0,
    "verbose": 2,
}

kwargs = {
    "n_bins": 255,
    "bin_subsample": 2e5,
    "max_layers": 10,
    "criterion": "mse",
    "n_estimators": 2,
    "n_trees": 100,
    "max_depth": None,
    "min_samples_leaf": 1,
    "use_predictor": True,
    "predictor": "forest",
    "predictor_kwargs": {},
    "n_tolerant_rounds": 2,
    "delta": 1e-5,
    "n_jobs": -1,
    "random_state": 0,
    "verbose": 2,
}


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            {"predictor_kwargs": {}, "n_job": 2},
            {"n_job": 2},
        ),
        (
            {"predictor_kwargs": {"n_job": 3}, "n_job": 2},
            {"n_job": 3},
        ),
        (
            {"predictor_kwargs": {"iter": 4}, "n_job": 2},
            {"iter": 4, "n_job": 2},
        ),
    ],
)
def test_predictor_kwargs_overwrite(test_input, expected):
    assert _get_predictor_kwargs(**test_input) == expected


def test_model_properties_after_fitting():
    """Check the model properties after fitting a deep forest model."""
    model = CascadeForestRegressor(**toy_kwargs)
    model.fit(X_train, y_train)

    assert len(model) == model.n_layers_

    assert model[0] is model._get_layer(0)

    with pytest.raises(ValueError) as excinfo:
        model._get_layer(model.n_layers_)
    assert "The layer index should be in the range" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        model._set_layer(0, None)
    assert "already exists in the internal container" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        model._get_binner(model.n_layers_ + 1)
    assert "The binner index should be in the range" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        model._set_binner(0, None)
    assert "already exists in the internal container" in str(excinfo.value)


def test_model_workflow_partial_mode():
    """Run the workflow of deep forest with a local buffer."""

    case_kwargs = copy.deepcopy(kwargs)
    case_kwargs.update({"partial_mode": True})

    model = CascadeForestRegressor(**case_kwargs)
    model.fit(X_train, y_train)

    # Predictions before saving
    y_pred_before = model.predict(X_test).astype(np.float32)

    # Save and Reload
    model.save(save_dir)

    model = CascadeForestRegressor(**case_kwargs)
    model.load(save_dir)

    # Predictions after loading
    y_pred_after = model.predict(X_test).astype(np.float32)

    # Make sure the same predictions before and after model serialization
    assert_array_equal(y_pred_before, y_pred_after)

    model.clean()  # clear the buffer
    shutil.rmtree(save_dir)


def test_model_workflow_in_memory():
    """Run the workflow of deep forest with in-memory mode."""

    case_kwargs = copy.deepcopy(kwargs)
    case_kwargs.update({"partial_mode": False})

    model = CascadeForestRegressor(**case_kwargs)
    model.fit(X_train, y_train)

    # Predictions before saving
    y_pred_before = model.predict(X_test).astype(np.float32)

    # Save and Reload
    model.save(save_dir)

    model = CascadeForestRegressor(**case_kwargs)
    model.load(save_dir)

    # Make sure the same predictions before and after model serialization
    y_pred_after = model.predict(X_test).astype(np.float32)

    assert_array_equal(y_pred_before, y_pred_after)

    shutil.rmtree(save_dir)


@pytest.mark.parametrize(
    "param",
    [
        (0, {"max_layers": 0}),
        (1, {"n_tolerant_rounds": 0}),
        (2, {"delta": -1}),
    ],
)
def test_model_invalid_training_params(param):
    case_kwargs = copy.deepcopy(toy_kwargs)
    case_kwargs.update(param[1])

    model = CascadeForestRegressor(**case_kwargs)

    with pytest.raises(ValueError) as excinfo:
        model.fit(X_train, y_train)

    if param[0] == 0:
        assert "max_layers" in str(excinfo.value)
    elif param[0] == 1:
        assert "n_tolerant_rounds" in str(excinfo.value)
    elif param[0] == 2:
        assert "delta " in str(excinfo.value)


@pytest.mark.parametrize("predictor", ["forest", "xgboost", "lightgbm"])
def test_regressor_predictor_normal(predictor):
    deepforest.cascade._build_regressor_predictor(
        predictor, criterion="mse", n_estimators=1, n_outputs=2
    )


def test_regressor_predictor_unknown():
    with pytest.raises(NotImplementedError) as excinfo:
        deepforest.cascade._build_regressor_predictor(
            "unknown", criterion="mse", n_estimators=1, n_outputs=2
        )
    assert "name of the predictor should be one of" in str(excinfo.value)


def test_model_n_trees_non_positive():
    case_kwargs = copy.deepcopy(toy_kwargs)
    case_kwargs.update({"n_trees": 0})
    model = CascadeForestRegressor(**case_kwargs)
    with pytest.raises(ValueError) as excinfo:
        model._set_n_trees(0)
    assert "should be strictly positive." in str(excinfo.value)


def test_model_n_trees_auto():
    case_kwargs = copy.deepcopy(toy_kwargs)
    case_kwargs.update({"n_trees": "auto"})
    model = CascadeForestRegressor(**case_kwargs)

    n_trees = model._set_n_trees(0)
    assert n_trees == 100

    n_trees = model._set_n_trees(2)
    assert n_trees == 300

    n_trees = model._set_n_trees(10)
    assert n_trees == 500


def test_model_n_trees_invalid():
    case_kwargs = copy.deepcopy(toy_kwargs)
    case_kwargs.update({"n_trees": [42]})
    model = CascadeForestRegressor(**case_kwargs)
    with pytest.raises(ValueError) as excinfo:
        model._set_n_trees(0)
    assert "Invalid value for n_trees." in str(excinfo.value)
