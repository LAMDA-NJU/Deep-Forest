import numpy as np
from numpy.testing import assert_array_equal

from sklearn.datasets import load_digits
from deepforest import CascadeForestClassifier


toy_kwargs = {
    "n_bins": 10,
    "bin_subsample": 2e5,
    "max_layers": 10,
    "n_estimators": 1,
    "criterion": "gini",
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


def test_model_input_label_encoder():
    """Test if the model behaves the same with and without label encoding."""

    # Load data
    X, y = load_digits(return_X_y=True)
    y_as_str = np.char.add("label_", y.astype(str))

    # Train model on integer labels. Labels should look like: 1, 2, 3, ...
    model = CascadeForestClassifier(**toy_kwargs)
    model.fit(X, y)
    y_pred_int_labels = model.predict(X)

    # Train model on string labels. Labels should look like: "label_1", "label_2", "label_3", ...
    model = CascadeForestClassifier(**toy_kwargs)
    model.fit(X, y_as_str)
    y_pred_str_labels = model.predict(X)

    # Check if the underlying data are the same
    y_pred_int_labels_as_str = np.char.add(
        "label_", y_pred_int_labels.astype(str)
    )
    assert_array_equal(y_pred_str_labels, y_pred_int_labels_as_str)

    # Clean up buffer
    model.clean()
