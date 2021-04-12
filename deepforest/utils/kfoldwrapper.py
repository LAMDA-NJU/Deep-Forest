"""
Implementation of the estimator wrapper to support customized base estimators.
"""


__all__ = ["KFoldWrapper"]

import copy
import numpy as np
from sklearn.model_selection import KFold

from .. import _utils


class KFoldWrapper(object):
    """
    A general wrapper for base estimators without the characteristic of
    out-of-bag (OOB) estimation.
    """

    def __init__(
        self,
        estimator,
        n_splits,
        n_outputs,
        random_state=None,
        verbose=1,
        is_classifier=True,
    ):

        # Parameters were already validated by upstream methods
        self.dummy_estimator_ = estimator
        self.n_splits = n_splits
        self.n_outputs = n_outputs
        self.random_state = random_state
        self.verbose = verbose
        self.is_classifier = is_classifier
        # Internal container
        self.estimators_ = []

    @property
    def estimator_(self):
        """Return the list of internal estimators."""
        return self.estimators_

    def fit_transform(self, X, y, sample_weight=None):
        n_samples, _ = X.shape
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        self.oob_decision_function_ = np.zeros((n_samples, self.n_outputs))

        for k, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            estimator = copy.deepcopy(self.dummy_estimator_)

            if self.verbose > 1:
                msg = "{} - - Fitting the base estimator with fold = {}"
                print(msg.format(_utils.ctime(), k))

            # Fit on training samples
            if sample_weight is None:
                # Notice that a bunch of base estimators do not take
                # `sample_weight` as a valid input.
                estimator.fit(X[train_idx], y[train_idx])
            else:
                estimator.fit(
                    X[train_idx], y[train_idx], sample_weight[train_idx]
                )

            # Predict on hold-out samples
            if self.is_classifier:
                self.oob_decision_function_[
                    val_idx
                ] += estimator.predict_proba(X[val_idx])
            else:
                val_pred = estimator.predict(X[val_idx])

                # Reshape for univariate regression
                if self.n_outputs == 1 and len(val_pred.shape) == 1:
                    val_pred = np.expand_dims(val_pred, 1)
                self.oob_decision_function_[val_idx] += val_pred

            # Store the estimator
            self.estimators_.append(estimator)

        return self.oob_decision_function_

    def predict(self, X):
        n_samples, _ = X.shape
        out = np.zeros((n_samples, self.n_outputs))  # pre-allocate results
        for estimator in self.estimators_:
            if self.is_classifier:
                out += estimator.predict_proba(X)  # classification
            else:
                if self.n_outputs > 1:
                    out += estimator.predict(X)  # multi-variate regression
                else:
                    out += estimator.predict(X).reshape(
                        n_samples, -1
                    )  # univariate regression

        return out / self.n_splits  # return the average prediction
