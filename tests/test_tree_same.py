"""
Testing cases here make sure that predictions of the reduced implementation
on decision tree is exactly the same as the original version in Scikit-Learn
after data binning.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.tree import (
    DecisionTreeClassifier as sklearn_DecisionTreeClassifier,
)
from sklearn.tree import (
    DecisionTreeRegressor as sklearn_DecisionTreeRegressor,
)
from sklearn.tree import ExtraTreeClassifier as sklearn_ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor as sklearn_ExtraTreeRegressor

# Load utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper

# Toy datasets
from sklearn.datasets import load_iris, load_wine, load_boston

from deepforest import DecisionTreeClassifier
from deepforest import ExtraTreeClassifier
from deepforest import DecisionTreeRegressor
from deepforest import ExtraTreeRegressor

test_size = 0.42
random_state = 42


@pytest.mark.parametrize("load_func", [load_iris, load_wine])
def test_tree_classifier_proba(load_func):

    X, y = load_func(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Data binning
    binner = _BinMapper(random_state=random_state)
    X_train_binned = binner.fit_transform(X_train)
    X_test_binned = binner.transform(X_test)

    # Ours
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train_binned, y_train)
    actual_pred = model.predict(X_test_binned)
    actual_proba = model.predict_proba(X_test_binned)

    # Sklearn
    model = sklearn_DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train_binned, y_train)
    expected_pred = model.predict(X_test_binned)
    expected_proba = model.predict_proba(X_test_binned)

    assert_array_equal(actual_pred, expected_pred)
    assert_array_equal(actual_proba, expected_proba)


@pytest.mark.parametrize("load_func", [load_iris, load_wine])
def test_extra_tree_classifier_proba(load_func):
    X, y = load_func(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Data binning
    binner = _BinMapper(random_state=random_state)
    X_train_binned = binner.fit_transform(X_train)
    X_test_binned = binner.transform(X_test)

    # Ours
    model = ExtraTreeClassifier(random_state=random_state)
    model.fit(X_train_binned, y_train)
    actual_pred = model.predict(X_test_binned)
    actual_proba = model.predict_proba(X_test_binned)

    # Sklearn
    model = sklearn_ExtraTreeClassifier(random_state=random_state)
    model.fit(X_train_binned, y_train)
    expected_pred = model.predict(X_test_binned)
    expected_proba = model.predict_proba(X_test_binned)

    assert_array_equal(actual_pred, expected_pred)
    assert_array_equal(actual_proba, expected_proba)


@pytest.mark.parametrize("load_func", [load_boston])
def test_tree_regressor_pred(load_func):

    X, y = load_func(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Data binning
    binner = _BinMapper(random_state=random_state)
    X_train_binned = binner.fit_transform(X_train)
    X_test_binned = binner.transform(X_test)

    # Ours
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train_binned, y_train)
    actual_pred = model.predict(X_test_binned)

    # Sklearn
    model = sklearn_DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train_binned, y_train)
    expected_pred = model.predict(X_test_binned)

    assert_array_equal(actual_pred, expected_pred)


@pytest.mark.parametrize("load_func", [load_boston])
def test_extra_tree_regressor_pred(load_func):
    X, y = load_func(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Data binning
    binner = _BinMapper(random_state=random_state)
    X_train_binned = binner.fit_transform(X_train)
    X_test_binned = binner.transform(X_test)

    # Ours
    model = ExtraTreeRegressor(random_state=random_state)
    model.fit(X_train_binned, y_train)
    actual_pred = model.predict(X_test_binned)

    # Sklearn
    model = sklearn_ExtraTreeRegressor(random_state=random_state)
    model.fit(X_train_binned, y_train)
    expected_pred = model.predict(X_test_binned)

    assert_array_equal(actual_pred, expected_pred)


@pytest.mark.parametrize("load_func", [load_boston])
def test_tree_regressor_multi_output_pred(load_func):

    X, y = load_func(return_X_y=True)

    # Generate pseudo multi output targets
    y = np.expand_dims(y, axis=1)
    y = np.concatenate((y, -y), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Data binning
    binner = _BinMapper(random_state=random_state)
    X_train_binned = binner.fit_transform(X_train)
    X_test_binned = binner.transform(X_test)

    # Ours
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train_binned, y_train)
    actual_pred = model.predict(X_test_binned)

    # Sklearn
    model = sklearn_DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train_binned, y_train)
    expected_pred = model.predict(X_test_binned)

    assert_array_equal(actual_pred, expected_pred)


@pytest.mark.parametrize("load_func", [load_boston])
def test_extra_tree_regressor_multi_output_pred(load_func):
    X, y = load_func(return_X_y=True)

    # Generate pseudo multi output targets
    y = np.expand_dims(y, axis=1)
    y = np.concatenate((y, -y), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Data binning
    binner = _BinMapper(random_state=random_state)
    X_train_binned = binner.fit_transform(X_train)
    X_test_binned = binner.transform(X_test)

    # Ours
    model = ExtraTreeRegressor(random_state=random_state)
    model.fit(X_train_binned, y_train)
    actual_pred = model.predict(X_test_binned)

    # Sklearn
    model = sklearn_ExtraTreeRegressor(random_state=random_state)
    model.fit(X_train_binned, y_train)
    expected_pred = model.predict(X_test_binned)

    assert_array_equal(actual_pred, expected_pred)
