import pytest
import shutil
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from deepforest import CascadeForestClassifier, CascadeForestRegressor


save_dir = "./tmp"

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.42, random_state=42
)


def test_custom_cascade_layer_workflow_in_memory():

    model = CascadeForestClassifier()

    n_estimators = 4
    estimators = [DecisionTreeClassifier() for _ in range(n_estimators)]
    model.set_estimator(estimators)  # set custom base estimators

    predictor = DecisionTreeClassifier()
    model.set_predictor(predictor)

    model.fit(X_train, y_train)
    y_pred_before = model.predict(X_test)

    # Save and Reload
    model.save(save_dir)

    model = CascadeForestClassifier()
    model.load(save_dir)

    # Predictions after loading
    y_pred_after = model.predict(X_test)

    # Make sure the same predictions before and after model serialization
    assert_array_equal(y_pred_before, y_pred_after)

    assert (
        model.get_estimator(0, 0, "custom")
        is model._get_layer(0).estimators_["0-0-custom"].estimator_
    )

    model.clean()  # clear the buffer
    shutil.rmtree(save_dir)


def test_custom_cascade_layer_workflow_partial_mode():

    model = CascadeForestClassifier(partial_mode=True)

    n_estimators = 4
    estimators = [DecisionTreeClassifier() for _ in range(n_estimators)]
    model.set_estimator(estimators)  # set custom base estimators

    predictor = DecisionTreeClassifier()
    model.set_predictor(predictor)

    model.fit(X_train, y_train)
    y_pred_before = model.predict(X_test)

    # Save and Reload
    model.save(save_dir)

    model = CascadeForestClassifier()
    model.load(save_dir)

    # Predictions after loading
    y_pred_after = model.predict(X_test)

    # Make sure the same predictions before and after model serialization
    assert_array_equal(y_pred_before, y_pred_after)

    model.clean()  # clear the buffer
    shutil.rmtree(save_dir)


def test_custom_base_estimator_wrong_estimator_type():

    model = CascadeForestClassifier()
    with pytest.raises(ValueError) as excinfo:
        model.set_estimator(42)
    assert "estimators should be a list" in str(excinfo.value)


def test_custom_estimator_missing_fit():
    class tmp_estimator:
        def __init__(self):
            pass

    model = CascadeForestClassifier()
    with pytest.raises(AttributeError) as excinfo:
        model.set_estimator([tmp_estimator()])
    assert "The `fit` method of estimator" in str(excinfo.value)

    with pytest.raises(AttributeError) as excinfo:
        model.set_predictor(tmp_estimator())
    assert "The `fit` method of the predictor" in str(excinfo.value)


def test_custom_base_estimator_missing_predict_proba():
    class tmp_estimator:
        def __init__(self):
            pass

        def fit(self, X, y):
            pass

    model = CascadeForestClassifier()
    with pytest.raises(AttributeError) as excinfo:
        model.set_estimator([tmp_estimator()])
    assert "The `predict_proba` method" in str(excinfo.value)

    with pytest.raises(AttributeError) as excinfo:
        model.set_predictor(tmp_estimator())
    assert "The `predict_proba` method of the predictor" in str(excinfo.value)


def test_custom_base_estimator_missing_predict():
    class tmp_estimator:
        def __init__(self):
            pass

        def fit(self, X, y):
            pass

    model = CascadeForestRegressor()
    with pytest.raises(AttributeError) as excinfo:
        model.set_estimator([tmp_estimator()])
    assert "The `predict` method" in str(excinfo.value)

    with pytest.raises(AttributeError) as excinfo:
        model.set_predictor(tmp_estimator())
    assert "The `predict` method of the predictor" in str(excinfo.value)


def test_custom_base_estimator_invalid_n_splits():

    model = CascadeForestRegressor()
    n_estimators = 4
    estimators = [DecisionTreeClassifier() for _ in range(n_estimators)]
    with pytest.raises(ValueError) as excinfo:
        model.set_estimator(estimators, n_splits=1)
    assert "should be at least 2" in str(excinfo.value)
