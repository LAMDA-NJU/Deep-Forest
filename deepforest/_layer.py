"""Implementation of the cascade layer in deep forest."""


__all__ = [
    "BaseCascadeLayer",
    "ClassificationCascadeLayer",
    "RegressionCascadeLayer",
    "CustomCascadeLayer",
]

import numpy as np
from sklearn.base import is_classifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from . import _utils
from ._estimator import Estimator
from .utils.kfoldwrapper import KFoldWrapper


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


class BaseCascadeLayer(BaseEstimator):
    def __init__(
        self,
        layer_idx,
        n_outputs,
        criterion,
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_leaf=1,
        backend="custom",
        partial_mode=False,
        buffer=None,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        self.layer_idx = layer_idx
        self.n_outputs = n_outputs
        self.criterion = criterion
        self.n_estimators = n_estimators * 2  # internal conversion
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.backend = backend
        self.partial_mode = partial_mode
        self.buffer = buffer
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        # Internal container
        self.estimators_ = {}

    @property
    def n_trees_(self):
        return self.n_estimators * self.n_trees

    @property
    def feature_importances_(self):
        feature_importances_ = np.zeros((self.n_features,))
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            # Partial mode
            if isinstance(estimator, str):
                estimator_ = self.buffer.load_estimator(estimator)
                feature_importances_ += estimator_.feature_importances_
            # In-memory mode
            else:
                feature_importances_ += estimator.feature_importances_

        return feature_importances_ / len(self.estimators_)

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
            backend=self.backend,
            n_jobs=self.n_jobs,
            random_state=random_state,
            is_classifier=is_classifier(self),
        )

        return estimator

    def _validate_params(self):

        if not self.n_estimators > 0:
            msg = "`n_estimators` = {} should be strictly positive."
            raise ValueError(msg.format(self.n_estimators))

        if not self.n_trees > 0:
            msg = "`n_trees` = {} should be strictly positive."
            raise ValueError(msg.format(self.n_trees))

    def transform(self, X):
        """Preserved for the naming consistency."""
        return self.predict_full(X)

    def predict_full(self, X):
        """Return the concatenated predictions from all base estimators."""
        n_samples, _ = X.shape
        pred = np.zeros((n_samples, self.n_outputs * self.n_estimators))
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            if self.verbose > 1:
                msg = "{} - Evaluating estimator = {:<5} in layer = {}"
                key = key.split("-")[-1] + "_" + str(key.split("-")[-2])
                print(msg.format(_utils.ctime(), key, self.layer_idx))
            if self.partial_mode:
                # Load the estimator from the buffer
                estimator = self.buffer.load_estimator(estimator)

            left, right = self.n_outputs * idx, self.n_outputs * (idx + 1)
            pred[:, left:right] += estimator.predict(X)

        return pred


class ClassificationCascadeLayer(BaseCascadeLayer, ClassifierMixin):
    """Implementation of the cascade forest layer for classification."""

    def __init__(
        self,
        layer_idx,
        n_outputs,
        criterion,
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_leaf=1,
        backend="custom",
        partial_mode=False,
        buffer=None,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        super().__init__(
            layer_idx=layer_idx,
            n_outputs=n_outputs,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            backend=backend,
            partial_mode=partial_mode,
            buffer=buffer,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def fit_transform(self, X, y, sample_weight=None):

        self._validate_params()
        n_samples, self.n_features = X.shape

        X_aug = []
        oob_decision_function = np.zeros((n_samples, self.n_outputs))

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
        y_pred = np.argmax(oob_decision_function, axis=1)
        self.val_performance_ = accuracy_score(
            y, y_pred, sample_weight=sample_weight
        )

        X_aug = np.hstack(X_aug)
        return X_aug


class RegressionCascadeLayer(BaseCascadeLayer, RegressorMixin):
    """Implementation of the cascade forest layer for regression."""

    def __init__(
        self,
        layer_idx,
        n_outputs,
        criterion,
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_leaf=1,
        backend="custom",
        partial_mode=False,
        buffer=None,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        super().__init__(
            layer_idx=layer_idx,
            n_outputs=n_outputs,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            backend=backend,
            partial_mode=partial_mode,
            buffer=buffer,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def fit_transform(self, X, y, sample_weight=None):

        self._validate_params()
        n_samples, self.n_features = X.shape

        X_aug = []
        oob_decision_function = np.zeros((n_samples, self.n_outputs))

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

        # Set the OOB estimations and validation mean squared error
        self.oob_decision_function_ = oob_decision_function / self.n_estimators
        y_pred = self.oob_decision_function_
        self.val_performance_ = mean_squared_error(
            y, y_pred, sample_weight=sample_weight
        )

        X_aug = np.hstack(X_aug)
        return X_aug


class CustomCascadeLayer(object):
    """Implementation of the cascade layer for customized base estimators."""

    def __init__(
        self,
        layer_idx,
        n_splits,
        n_outputs,
        estimators,
        partial_mode=False,
        buffer=None,
        random_state=None,
        verbose=1,
        is_classifier=True,
    ):
        self.layer_idx = layer_idx
        self.n_splits = n_splits
        self.n_outputs = n_outputs
        self.n_estimators = len(estimators)
        self.dummy_estimators_ = estimators
        self.partial_mode = partial_mode
        self.buffer = buffer
        self.random_state = random_state
        self.verbose = verbose
        self.is_classifier = is_classifier
        # Internal container
        self.estimators_ = {}

    def fit_transform(self, X, y, sample_weight=None):
        n_samples, _ = X.shape
        X_aug = []

        # Parameters were already validated by upstream methods
        for estimator_idx, estimator in enumerate(self.dummy_estimators_):
            kfold_estimator = KFoldWrapper(
                estimator,
                self.n_splits,
                self.n_outputs,
                self.random_state,
                self.verbose,
                self.is_classifier,
            )

            if self.verbose > 1:
                msg = "{} - Fitting estimator = custom_{} in layer = {}"
                print(
                    msg.format(_utils.ctime(), estimator_idx, self.layer_idx)
                )

            kfold_estimator.fit_transform(X, y, sample_weight)
            X_aug.append(kfold_estimator.oob_decision_function_)
            key = "{}-{}-custom".format(self.layer_idx, estimator_idx)

            if self.partial_mode:
                # Cache the fitted estimator in out-of-core mode
                buffer_path = self.buffer.cache_estimator(
                    self.layer_idx, estimator_idx, "custom", kfold_estimator
                )
                self.estimators_.update({key: buffer_path})
            else:
                self.estimators_.update({key: kfold_estimator})

        # Set the OOB estimations and validation performance
        oob_decision_function = np.zeros_like(X_aug[0])
        for estimator_oob_decision_function in X_aug:
            oob_decision_function += (
                estimator_oob_decision_function / self.n_estimators
            )

        if self.is_classifier:  # classification
            y_pred = np.argmax(oob_decision_function, axis=1)
            self.val_performance_ = accuracy_score(
                y, y_pred, sample_weight=sample_weight
            )
        else:  # regression
            self.val_performance_ = mean_squared_error(
                y, y_pred, sample_weight=sample_weight
            )

        X_aug = np.hstack(X_aug)
        return X_aug

    def transform(self, X):
        """Preserved for the naming consistency."""
        return self.predict_full(X)

    def predict_full(self, X):
        """Return the concatenated predictions from all base estimators."""
        n_samples, _ = X.shape
        pred = np.zeros((n_samples, self.n_outputs * self.n_estimators))
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            if self.verbose > 1:
                msg = "{} - Evaluating estimator = custom_{} in layer = {}"
                print(msg.format(_utils.ctime(), idx, self.layer_idx))
            if self.partial_mode:
                # Load the estimator from the buffer
                estimator = self.buffer.load_estimator(estimator)

            left, right = self.n_outputs * idx, self.n_outputs * (idx + 1)
            pred[:, left:right] += estimator.predict(X)

        return pred
