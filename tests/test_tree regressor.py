import pytest
from sklearn.datasets import load_diabetes
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper

from deepforest import DecisionTreeRegressor


X, y = load_diabetes(return_X_y=True)

# Data binning
binner = _BinMapper(random_state=42)
X_binned = binner.fit_transform(X)


def test_tree_properties_after_fitting():
    tree = DecisionTreeRegressor()
    tree.fit(X_binned, y)

    assert tree.get_depth() == tree.tree_.max_depth
    assert tree.n_leaves == tree.tree_.n_leaves
    assert tree.n_internals == tree.tree_.n_internals


def test_tree_fit_invalid_dtype():
    tree = DecisionTreeRegressor()

    with pytest.raises(RuntimeError) as execinfo:
        tree.fit(X, y)
    assert "The dtype of `X` should be `np.uint8`" in str(execinfo.value)


def test_tree_fit_invalid_training_params():
    tree = DecisionTreeRegressor(min_samples_leaf=0)
    with pytest.raises(ValueError) as execinfo:
        tree.fit(X_binned, y)
    assert "min_samples_leaf must be at least 1" in str(execinfo.value)

    tree = DecisionTreeRegressor(min_samples_leaf=0.6)
    with pytest.raises(ValueError) as execinfo:
        tree.fit(X_binned, y)
    assert "or in (0, 0.5]" in str(execinfo.value)

    tree = DecisionTreeRegressor(min_samples_split=1)
    with pytest.raises(ValueError) as execinfo:
        tree.fit(X_binned, y)
    assert "min_samples_split must be an integer" in str(execinfo.value)

    tree = DecisionTreeRegressor(max_features="unknown")
    with pytest.raises(ValueError) as execinfo:
        tree.fit(X_binned, y)
    assert "Invalid value for max_features." in str(execinfo.value)

    tree = DecisionTreeRegressor()
    with pytest.raises(ValueError) as execinfo:
        tree.fit(X_binned, y[:1])
    assert "Number of labels=" in str(execinfo.value)

    tree = DecisionTreeRegressor(min_weight_fraction_leaf=0.6)
    with pytest.raises(ValueError) as execinfo:
        tree.fit(X_binned, y)
    assert "min_weight_fraction_leaf must in [0, 0.5]" in str(execinfo.value)

    tree = DecisionTreeRegressor(max_depth=0)
    with pytest.raises(ValueError) as execinfo:
        tree.fit(X_binned, y)
    assert "max_depth must be greater than zero." in str(execinfo.value)
