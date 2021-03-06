import shutil
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from deepforest import CascadeForestClassifier


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
    model.set_estimators(estimators)  # set custom base estimators

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


def test_custom_cascade_layer_workflow_partial_mode():

    model = CascadeForestClassifier(partial_mode=True)

    n_estimators = 4
    estimators = [DecisionTreeClassifier() for _ in range(n_estimators)]
    model.set_estimators(estimators)  # set custom base estimators

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


if __name__ == "__main__":

    test_custom_cascade_layer_workflow_partial_mode()
