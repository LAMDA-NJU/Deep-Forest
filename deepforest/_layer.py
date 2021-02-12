"""Implementation of the forest-based cascade layer."""


__all__ = ["Layer"]

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from . import _utils
from ._estimator import Estimator


def _build_estimator(
    X,
    y,
    layer_idx,
    estimator_idx,
    estimator_name,
    estimator,
    oob_decision_function,
    partial_mode=True,
    buffer=None,
    verbose=1,
    sample_weight=None,
):
    """Private function used to fit a single estimator."""
    if verbose > 1:
        msg = "{} - Fitting estimator = {:<5} in layer = {}"
        key = estimator_name + "_" + str(estimator_idx)
        print(msg.format(_utils.ctime(), key, layer_idx))

    X_aug_train = estimator.fit_transform(X, y, sample_weight)
    oob_decision_function += estimator.oob_decision_function_

    if partial_mode:
        # Cache the fitted estimator in out-of-core mode
        buffer_path = buffer.cache_estimator(
            layer_idx, estimator_idx, estimator_name, estimator
        )
        return X_aug_train, buffer_path
    else:
        return X_aug_train, estimator


class Layer(object):
    def __init__(
        self,
        layer_idx,
        n_classes,
        criterion,
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_leaf=1,
        partial_mode=False,
        buffer=None,
        n_jobs=None,
        random_state=None,
        verbose=1,
        is_classifier=True,
    ):
        self.layer_idx = layer_idx
        self.n_classes = n_classes
        self.criterion = criterion
        self.n_estimators = n_estimators * 2  # internal conversion
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.partial_mode = partial_mode
        self.buffer = buffer
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.is_classifier = is_classifier
        # Internal container
        self.estimators_ = {}

    @property
    def n_trees_(self):
        return self.n_estimators * self.n_trees

    def _make_estimator(self, estimator_idx, estimator_name):
        """Make and configure a copy of the estimator."""
        # Set the non-overlapped random state
        if self.random_state is not None:
            random_state = (
                self.random_state + 10 * estimator_idx + 100 * self.layer_idx
            )
        else:
            random_state = None

        estimator = Estimator(
            name=estimator_name,
            criterion=self.criterion,
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=random_state,
            is_classifier=self.is_classifier,
        )

        return estimator

    def _validate_params(self):

        if not self.n_estimators > 0:
            msg = "`n_estimators` = {} should be strictly positive."
            raise ValueError(msg.format(self.n_estimators))

        if not self.n_trees > 0:
            msg = "`n_trees` = {} should be strictly positive."
            raise ValueError(msg.format(self.n_trees))

    def fit_transform(self, X, y, sample_weight=None):

        self._validate_params()
        n_samples, _ = X.shape

        X_aug = []
        if self.is_classifier:
            oob_decision_function = np.zeros((n_samples, self.n_classes))
        else:
            oob_decision_function = np.zeros((n_samples, 1))

        # A random forest and an extremely random forest will be fitted
        for estimator_idx in range(self.n_estimators // 2):
            X_aug_, _estimator = _build_estimator(
                X,
                y,
                self.layer_idx,
                estimator_idx,
                "rf",
                self._make_estimator(estimator_idx, "rf"),
                oob_decision_function,
                self.partial_mode,
                self.buffer,
                self.verbose,
                sample_weight,
            )
            X_aug.append(X_aug_)
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "rf")
            self.estimators_.update({key: _estimator})

        for estimator_idx in range(self.n_estimators // 2):
            X_aug_, _estimator = _build_estimator(
                X,
                y,
                self.layer_idx,
                estimator_idx,
                "erf",
                self._make_estimator(estimator_idx, "erf"),
                oob_decision_function,
                self.partial_mode,
                self.buffer,
                self.verbose,
                sample_weight,
            )
            X_aug.append(X_aug_)
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "erf")
            self.estimators_.update({key: _estimator})

        # Set the OOB estimations and validation accuracy
        self.oob_decision_function_ = oob_decision_function / self.n_estimators
        if self.is_classifier:
            y_pred = np.argmax(oob_decision_function, axis=1)
            self.val_acc_ = accuracy_score(
                y, y_pred, sample_weight=sample_weight
            )
        else:
            y_pred = self.oob_decision_function_
            self.val_acc_ = mean_squared_error(
                y, y_pred, sample_weight=sample_weight
            )

        X_aug = np.hstack(X_aug)
        return X_aug

    def transform(self, X, is_classifier):
        """
        Return the concatenated transformation results from all base
        estimators."""
        n_samples, _ = X.shape
        if is_classifier:
            X_aug = np.zeros((n_samples, self.n_classes * self.n_estimators))
        else:
            X_aug = np.zeros((n_samples, self.n_estimators))
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            if self.verbose > 1:
                msg = "{} - Evaluating estimator = {:<5} in layer = {}"
                key = key.split("-")[-1] + "_" + str(key.split("-")[-2])
                print(msg.format(_utils.ctime(), key, self.layer_idx))
            if self.partial_mode:
                # Load the estimator from the buffer
                estimator = self.buffer.load_estimator(estimator)

            if is_classifier:
                left, right = self.n_classes * idx, self.n_classes * (idx + 1)
            else:
                left, right = idx, (idx + 1)
            X_aug[:, left:right] += estimator.predict(X)

        return X_aug

    def predict_full(self, X, is_classifier):
        """Return the concatenated predictions from all base estimators."""
        n_samples, _ = X.shape
        if is_classifier:
            pred = np.zeros((n_samples, self.n_classes * self.n_estimators))
        else:
            pred = np.zeros((n_samples, self.n_estimators))
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            if self.verbose > 1:
                msg = "{} - Evaluating estimator = {:<5} in layer = {}"
                key = key.split("-")[-1] + "_" + str(key.split("-")[-2])
                print(msg.format(_utils.ctime(), key, self.layer_idx))
            if self.partial_mode:
                # Load the estimator from the buffer
                estimator = self.buffer.load_estimator(estimator)

            if is_classifier:
                left, right = self.n_classes * idx, self.n_classes * (idx + 1)
            else:
                left, right = idx, (idx + 1)
            pred[:, left:right] += estimator.predict(X)

        return pred
