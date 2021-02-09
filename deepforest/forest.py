"""
Implementation of the forest model for classification in Deep Forest.

This class is modified from:
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py
"""


__all__ = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
]

import numbers
from warnings import warn
import threading
from typing import List

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from joblib import Parallel, delayed
from joblib import effective_n_jobs

from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import is_classifier
from sklearn.base import ClassifierMixin, RegressorMixin, MultiOutputMixin
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.validation import _deprecate_positional_args

from . import _cutils as _LIB
from . import _forest as _C_FOREST

from .tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from .tree._tree import DOUBLE


MAX_INT = np.iinfo(np.int32).max


def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0, 1)`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.

    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples < 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return int(round(n_samples * max_samples))

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _generate_sample_mask(random_state, n_samples, n_samples_bootstrap):
    """Private function used to _parallel_build_trees function."""

    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
    sample_indices = sample_indices.astype(np.int32)
    sample_mask = _LIB._c_sample_mask(sample_indices, n_samples)

    return sample_mask


def _parallel_build_trees(
    tree,
    X,
    y,
    n_samples_bootstrap,
    sample_weight,
    out,
    mask,
    is_classifier,
    lock,
):
    """
    Private function used to fit a single tree in parallel."""
    n_samples = X.shape[0]

    sample_mask = _generate_sample_mask(
        tree.random_state, n_samples, n_samples_bootstrap
    )

    # Fit the tree on the bootstrapped samples
    if sample_weight is not None:
        sample_weight = sample_weight[sample_mask]
    feature, threshold, children, value = tree.fit(
        X[sample_mask],
        y[sample_mask],
        sample_weight=sample_weight,
        check_input=False,
    )

    if not children.flags["C_CONTIGUOUS"]:
        children = np.ascontiguousarray(children)

    if not value.flags["C_CONTIGUOUS"]:
        value = np.ascontiguousarray(value)

    value = np.squeeze(value, axis=1)

    if is_classifier:
        value /= value.sum(axis=1)[:, np.newaxis]

    # Set the OOB predictions
    oob_prediction = _C_FOREST.predict(
        X[~sample_mask, :], feature, threshold, children, value
    )
    with lock:

        mask += ~sample_mask
        out[~sample_mask, :] += oob_prediction

    return feature, threshold, children, value


# [Source] Sklearn.ensemble._base.py
def _set_random_states(estimator, random_state=None):
    """Set fixed random_state parameters for an estimator.

    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.

    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.

    random_state : int or RandomState, default=None
        Pseudo-random number generator to control the generation of the random
        integers. Pass an int for reproducible output across multiple function
        calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:

        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)


# [Source] Sklearn.ensemble._base.py
def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(
        n_jobs, n_estimators // n_jobs, dtype=np.int
    )
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _accumulate_prediction(feature, threshold, children, value, X, out, lock):
    """This is a utility function for joblib's Parallel."""
    prediction = _C_FOREST.predict(X, feature, threshold, children, value)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


# [Source] Sklearn.ensemble._base.py
class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for all ensemble classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    base_estimator : object
        The base estimator from which the ensemble is built.

    n_estimators : int, default=10
        The number of estimators in the ensemble.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    # overwrite _required_parameters from MetaEstimatorMixin
    _required_parameters: List[str] = []

    @abstractmethod
    def __init__(
        self, base_estimator, *, n_estimators=10, estimator_params=tuple()
    ):
        # Set parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        # Don't instantiate estimators now! Parameters of base_estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # self.estimators_ needs to be filled by the derived classes in fit.

    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute.

        Sets the base_estimator_` attributes.
        """
        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError(
                "n_estimators must be an integer, "
                "got {0}.".format(type(self.n_estimators))
            )

        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators must be greater than zero, "
                "got {0}.".format(self.n_estimators)
            )

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(
            **{p: getattr(self, p) for p in self.estimator_params}
        )

        # Pass the inferred class information to avoid redudant finding.
        if is_classifier(estimator):
            estimator.classes_ = self.classes_
            estimator.n_classes_ = np.array(self.n_classes_, dtype=np.int32)

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Return the index'th estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        """Return iterator over estimators in the ensemble."""
        return iter(self.estimators_)


class BaseForest(MultiOutputMixin, BaseEnsemble, metaclass=ABCMeta):
    """
    Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        n_jobs=None,
        random_state=None,
        verbose=0,
        class_weight=None,
        max_samples=None
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
        )

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.max_samples = max_samples

        # Internal containers
        self.features = []
        self.thresholds = []
        self.childrens = []
        self.values = []

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
        """
        # Validate or convert input data
        if issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]
        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0], max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()
        random_state = check_random_state(self.random_state)
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        trees = [
            self._make_estimator(append=False, random_state=random_state)
            for i in range(self.n_estimators)
        ]

        # Pre-allocate OOB estimations
        if is_classifier(self):
            oob_decision_function = np.zeros(
                (n_samples, self.classes_[0].shape[0])
            )
        else:
            oob_decision_function = np.zeros((n_samples, 1))
        mask = np.zeros(n_samples)

        lock = threading.Lock()
        rets = Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer="threads", require="sharedmem")
        )(
            delayed(_parallel_build_trees)(
                t,
                X,
                y,
                n_samples_bootstrap,
                sample_weight,
                oob_decision_function,
                mask,
                is_classifier(self),
                lock,
            )
            for i, t in enumerate(trees)
        )
        # Collect newly grown trees
        for feature, threshold, children, value in rets:

            # No check on feature and threshold since 1-D array is always
            # C-aligned and F-aligned.
            self.features.append(feature)
            self.thresholds.append(threshold)
            self.childrens.append(children)
            self.values.append(value)

        # Check the OOB predictions
        if (
            is_classifier(self)
            and (oob_decision_function.sum(axis=1) == 0).any()
        ):
            warn(
                "Some inputs do not have OOB predictions. "
                "This probably means too few trees were used "
                "to compute any reliable oob predictions."
            )
        if is_classifier(self):
            prediction = (
                oob_decision_function
                / oob_decision_function.sum(axis=1)[:, np.newaxis]
            )
        else:
            prediction = oob_decision_function / mask.reshape(-1, 1)

        self.oob_decision_function_ = prediction

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def _validate_y_class_weight(self, y):
        # Default implementation
        return y, None


class ForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
    """
    Base class for forest of trees-based classifiers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        n_jobs=None,
        random_state=None,
        verbose=0,
        class_weight=None,
        max_samples=None
    ):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            class_weight=class_weight,
            max_samples=max_samples,
        )

    def _validate_y_class_weight(self, y):

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(
                y[:, k], return_inverse=True
            )
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        if self.class_weight is not None:
            valid_presets = ("balanced", "balanced_subsample")
            if isinstance(self.class_weight, str):
                if self.class_weight not in valid_presets:
                    raise ValueError(
                        "Valid presets for class_weight include "
                        '"balanced" and "balanced_subsample".'
                        'Given "%s".' % self.class_weight
                    )

            if self.class_weight != "balanced_subsample" or not self.bootstrap:
                if self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(
                    class_weight, y_original
                )

        return y, expanded_class_weight

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        check_is_fitted(self)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Avoid storing the output of every estimator by summing them here
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]
        lock = threading.Lock()
        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(require="sharedmem")
        )(
            delayed(_accumulate_prediction)(
                self.features[i],
                self.thresholds[i],
                self.childrens[i],
                self.values[i],
                X,
                all_proba,
                lock,
            )
            for i in range(self.n_estimators)
        )

        for proba in all_proba:
            proba /= len(self.features)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba


class RandomForestClassifier(ForestClassifier):
    @_deprecate_positional_args
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        n_jobs=None,
        random_state=None,
        verbose=0,
        class_weight=None,
        max_samples=None
    ):
        super().__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "min_impurity_decrease",
                "min_impurity_split",
                "random_state",
            ),
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class ExtraTreesClassifier(ForestClassifier):
    @_deprecate_positional_args
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        n_jobs=None,
        random_state=None,
        verbose=0,
        class_weight=None,
        max_samples=None
    ):
        super().__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "min_impurity_decrease",
                "min_impurity_split",
                "random_state",
            ),
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class ForestRegressor(RegressorMixin, BaseForest, metaclass=ABCMeta):
    """
    Base class for forest of trees-based regressors.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        n_jobs=None,
        random_state=None,
        verbose=0,
        max_samples=None
    ):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            max_samples=max_samples,
        )

    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        y_hat = np.zeros((X.shape[0], 1), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(require="sharedmem")
        )(
            delayed(_accumulate_prediction)(
                self.features[i],
                self.thresholds[i],
                self.childrens[i],
                self.values[i],
                X,
                [y_hat],
                lock,
            )
            for i in range(self.n_estimators)
        )

        y_hat /= self.n_estimators
        return y_hat

    @staticmethod
    def _get_oob_predictions(tree, X):
        """Compute the OOB predictions for an individual tree.

        Parameters
        ----------
        tree : DecisionTreeRegressor object
            A single decision tree regressor.
        X : ndarray of shape (n_samples, n_features)
            The OOB samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, 1, n_outputs)
            The OOB associated predictions.
        """
        y_pred = tree.predict(X, check_input=False)
        if y_pred.ndim == 1:
            # single output regression
            y_pred = y_pred[:, np.newaxis, np.newaxis]
        else:
            # multioutput regression
            y_pred = y_pred[:, np.newaxis, :]
        return y_pred


class RandomForestRegressor(ForestRegressor):
    @_deprecate_positional_args
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        n_jobs=None,
        random_state=None,
        verbose=0,
        max_samples=None
    ):
        super().__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "min_impurity_decrease",
                "min_impurity_split",
                "random_state",
            ),
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class ExtraTreesRegressor(ForestRegressor):
    @_deprecate_positional_args
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        n_jobs=None,
        random_state=None,
        verbose=0,
        max_samples=None
    ):
        super().__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "min_impurity_decrease",
                "min_impurity_split",
                "random_state",
            ),
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
