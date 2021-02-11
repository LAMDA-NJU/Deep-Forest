"""Implementation of Deep Forest."""


__all__ = ["CascadeForestClassifier", "CascadeForestRegressor"]

import time
import numbers
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import is_classifier

from . import _utils
from . import _io
from ._layer import Layer
from ._binner import Binner


def _get_predictor_kwargs(predictor_kwargs, **kwargs) -> dict:
    """Overwrites default args if predictor_kwargs is supplied."""
    for key, value in kwargs.items():
        if key not in predictor_kwargs.keys():
            predictor_kwargs[key] = value
    return predictor_kwargs


def _build_classifier_predictor(
    predictor_name,
    criterion,
    n_estimators,
    n_outputs,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=None,
    random_state=None,
    predictor_kwargs={},
):
    """Build the predictor concatenated to the deep forest."""
    predictor_name = predictor_name.lower()

    # Random Forest
    if predictor_name == "forest":
        from .forest import RandomForestClassifier

        predictor = RandomForestClassifier(
            **_get_predictor_kwargs(
                predictor_kwargs,
                criterion=criterion,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    # XGBoost
    elif predictor_name == "xgboost":
        try:
            xgb = __import__("xgboost.sklearn")
        except ModuleNotFoundError:
            msg = (
                "Cannot load the module XGBoost when building the predictor."
                " Please make sure that XGBoost is installed."
            )
            raise ModuleNotFoundError(msg)

        # The argument `tree_method` is always set as `hist` for XGBoost,
        # because the exact mode of XGBoost is too slow.
        objective = "multi:softmax" if n_outputs > 2 else "binary:logistic"
        predictor = xgb.sklearn.XGBClassifier(
            **_get_predictor_kwargs(
                predictor_kwargs,
                objective=objective,
                n_estimators=n_estimators,
                tree_method="hist",
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    # LightGBM
    elif predictor_name == "lightgbm":
        try:
            lgb = __import__("lightgbm.sklearn")
        except ModuleNotFoundError:
            msg = (
                "Cannot load the module LightGBM when building the predictor."
                " Please make sure that LightGBM is installed."
            )
            raise ModuleNotFoundError(msg)

        objective = "multiclass" if n_outputs > 2 else "binary"
        predictor = lgb.LGBMClassifier(
            **_get_predictor_kwargs(
                predictor_kwargs,
                objective=objective,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    else:
        msg = (
            "The name of the predictor should be one of {{forest, xgboost,"
            " lightgbm}}, but got {} instead."
        )
        raise NotImplementedError(msg.format(predictor_name))

    return predictor


def _build_regressor_predictor(
    predictor_name,
    criterion,
    n_estimators,
    n_outputs,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=None,
    random_state=None,
    predictor_kwargs={},
):
    """Build the predictor concatenated to the deep forest."""
    predictor_name = predictor_name.lower()

    # Random Forest
    if predictor_name == "forest":
        from .forest import RandomForestRegressor

        predictor = RandomForestRegressor(
            **_get_predictor_kwargs(
                predictor_kwargs,
                criterion=criterion,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    # XGBoost
    elif predictor_name == "xgboost":
        try:
            xgb = __import__("xgboost.sklearn")
        except ModuleNotFoundError:
            msg = (
                "Cannot load the module XGBoost when building the predictor."
                " Please make sure that XGBoost is installed."
            )
            raise ModuleNotFoundError(msg)

        # The argument `tree_method` is always set as `hist` for XGBoost,
        # because the exact mode of XGBoost is too slow.
        objective = "reg:squarederror"
        predictor = xgb.sklearn.XGBRegressor(
            **_get_predictor_kwargs(
                predictor_kwargs,
                objective=objective,
                n_estimators=n_estimators,
                tree_method="hist",
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    # LightGBM
    elif predictor_name == "lightgbm":
        try:
            lgb = __import__("lightgbm.sklearn")
        except ModuleNotFoundError:
            msg = (
                "Cannot load the module LightGBM when building the predictor."
                " Please make sure that LightGBM is installed."
            )
            raise ModuleNotFoundError(msg)

        objective = "regression"
        predictor = lgb.LGBMRegressor(
            **_get_predictor_kwargs(
                predictor_kwargs,
                objective=objective,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    else:
        msg = (
            "The name of the predictor should be one of {{forest, xgboost,"
            " lightgbm}}, but got {} instead."
        )
        raise NotImplementedError(msg.format(predictor_name))

    return predictor


__classifier_model_doc = """
    Parameters
    ----------
    n_bins : :obj:`int`, default=255
        The number of bins used for non-missing values. In addition to the
        ``n_bins`` bins, one more bin is reserved for missing values. Its
        value must be no smaller than 2 and no greater than 255.
    bin_subsample : :obj:`int`, default=2e5
        The number of samples used to construct feature discrete bins. If
        the size of training set is smaller than ``bin_subsample``, then all
        training samples will be used.
    max_layers : :obj:`int`, default=20
        The maximum number of cascade layers in the deep forest. Notice that
        the actual number of layers can be smaller than ``max_layers`` because
        of the internal early stopping stage.
    criterion : :obj:`{"gini", "entropy"}`, default="gini"
        The function to measure the quality of a split. Supported criteria 
        are ``gini`` for the Gini impurity and ``entropy`` for the information 
        gain. Note: this parameter is tree-specific.
    n_estimators : :obj:`int`, default=2
        The number of estimator in each cascade layer. It will be multiplied
        by 2 internally because each estimator contains a
        :class:`RandomForestClassifier` and a :class:`ExtraTreesClassifier`,
        respectively.
    n_trees : :obj:`int`, default=100
        The number of trees in each estimator.
    max_depth : :obj:`int`, default=None
        The maximum depth of each tree. ``None`` indicates no constraint.
    min_samples_leaf : :obj:`int`, default=1
        The minimum number of samples required to be at a leaf node.
    use_predictor : :obj:`bool`, default=False
        Whether to build the predictor concatenated to the deep forest. Using
        the predictor may improve the performance of deep forest.
    predictor : :obj:`{"forest", "xgboost", "lightgbm"}`, default="forest"
        The type of the predictor concatenated to the deep forest. If
        ``use_predictor`` is False, this parameter will have no effect.
    predictor_kwargs : :obj:`dict`, default={}
        The configuration of the predictor concatenated to the deep forest.
        Specifying this will extend/overwrite the original parameters inherit
        from deep forest. If ``use_predictor`` is False, this parameter will
        have no effect.
    n_tolerant_rounds : :obj:`int`, default=2
        Specify when to conduct early stopping. The training process
        terminates when the validation performance on the training set does
        not improve compared against the best validation performance achieved
        so far for ``n_tolerant_rounds`` rounds.
    delta : :obj:`float`, default=1e-5
        Specify the threshold on early stopping. The counting on
        ``n_tolerant_rounds`` is triggered if the performance of a fitted
        cascade layer does not improve by ``delta`` compared against the best
        validation performance achieved so far.
    partial_mode : :obj:`bool`, default=False
        Whether to train the deep forest in partial mode. For large
        datasets, it is recommended to use the partial mode.

        - If ``True``, the partial mode is activated and all fitted
          estimators will be dumped in a local buffer;
        - If ``False``, all fitted estimators are directly stored in the
          memory.
    n_jobs : :obj:`int` or ``None``, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. None means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    random_state : :obj:`int` or ``None``, default=None

        - If :obj:`int`, ``random_state`` is the seed used by the random
          number generator;
        - If ``None``, the random number generator is the RandomState
          instance used by :mod:`np.random`.
    verbose : :obj:`int`, default=1
        Controls the verbosity when fitting and predicting.

        - If ``<= 0``, silent mode, which means no logging information will
          be displayed;
        - If ``1``, logging information on the cascade layer level will be
          displayed;
        - If ``> 1``, full logging information will be displayed.
"""


__classifier_fit_doc = """

    .. note::

        Deep forest supports two kinds of modes for training:

        - **Full memory mode**, in which the training / testing data and
          all fitted estimators are directly stored in the memory.
        - **Partial mode**, in which after fitting each estimator using
          the training data, it will be dumped in the buffer. During the
          evaluating stage, the dumped estimators are reloaded into the
          memory sequentially to evaluate the testing data.

        By setting the ``partial_mode`` to ``True``, the partial mode is
        activated, and a local buffer will be created at the current
        directory. The partial mode is able to reduce the running memory
        cost when training the deep forest.

    Parameters
    ----------
    X : :obj:`numpy.ndarray` of shape (n_samples, n_features)
        The training data. Internally, it will be converted to
        ``np.uint8``.
    y : :obj:`numpy.ndarray` of shape (n_samples,)
        The class labels of input samples.
    sample_weight : :obj:`numpy.ndarray` of shape (n_samples,), default=None
        Sample weights. If ``None``, then samples are equally weighted.
"""

__regressor_model_doc = """
    Parameters
    ----------
    n_bins : :obj:`int`, default=255
        The number of bins used for non-missing values. In addition to the
        ``n_bins`` bins, one more bin is reserved for missing values. Its
        value must be no smaller than 2 and no greater than 255.
    bin_subsample : :obj:`int`, default=2e5
        The number of samples used to construct feature discrete bins. If
        the size of training set is smaller than ``bin_subsample``, then all
        training samples will be used.
    max_layers : :obj:`int`, default=20
        The maximum number of cascade layers in the deep forest. Notice that
        the actual number of layers can be smaller than ``max_layers`` because
        of the internal early stopping stage.
    criterion : :obj:`{"mse", "mae"}`, default="mse"
        The function to measure the quality of a split. Supported criteria are 
        ``mse`` for the mean squared error, which is equal to variance reduction 
        as feature selection criterion, and ``mae`` for the mean absolute error.
    n_estimators : :obj:`int`, default=2
        The number of estimator in each cascade layer. It will be multiplied
        by 2 internally because each estimator contains a
        :class:`RandomForestRegressor` and a :class:`ExtraTreesRegressor`,
        respectively.
    n_trees : :obj:`int`, default=100
        The number of trees in each estimator.
    max_depth : :obj:`int`, default=None
        The maximum depth of each tree. ``None`` indicates no constraint.
    min_samples_leaf : :obj:`int`, default=1
        The minimum number of samples required to be at a leaf node.
    use_predictor : :obj:`bool`, default=False
        Whether to build the predictor concatenated to the deep forest. Using
        the predictor may improve the performance of deep forest.
    predictor : :obj:`{"forest", "xgboost", "lightgbm"}`, default="forest"
        The type of the predictor concatenated to the deep forest. If
        ``use_predictor`` is False, this parameter will have no effect.
    predictor_kwargs : :obj:`dict`, default={}
        The configuration of the predictor concatenated to the deep forest.
        Specifying this will extend/overwrite the original parameters inherit
        from deep forest.
        If ``use_predictor`` is False, this parameter will have no effect.
    n_tolerant_rounds : :obj:`int`, default=2
        Specify when to conduct early stopping. The training process
        terminates when the validation performance on the training set does
        not improve compared against the best validation performance achieved
        so far for ``n_tolerant_rounds`` rounds.
    delta : :obj:`float`, default=1e-5
        Specify the threshold on early stopping. The counting on
        ``n_tolerant_rounds`` is triggered if the performance of a fitted
        cascade layer does not improve by ``delta`` compared against the best
        validation performance achieved so far.
    partial_mode : :obj:`bool`, default=False
        Whether to train the deep forest in partial mode. For large
        datasets, it is recommended to use the partial mode.

        - If ``True``, the partial mode is activated and all fitted
          estimators will be dumped in a local buffer;
        - If ``False``, all fitted estimators are directly stored in the
          memory.
    n_jobs : :obj:`int` or ``None``, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. None means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    random_state : :obj:`int` or ``None``, default=None

        - If :obj:`int`, ``random_state`` is the seed used by the random
          number generator;
        - If ``None``, the random number generator is the RandomState
          instance used by :mod:`np.random`.
    verbose : :obj:`int`, default=1
        Controls the verbosity when fitting and predicting.

        - If ``<= 0``, silent mode, which means no logging information will
          be displayed;
        - If ``1``, logging information on the cascade layer level will be
          displayed;
        - If ``> 1``, full logging information will be displayed.
"""

__regressor_fit_doc = """

    .. note::

        Deep forest supports two kinds of modes for training:

        - **Full memory mode**, in which the training / testing data and
          all fitted estimators are directly stored in the memory.
        - **Partial mode**, in which after fitting each estimator using
          the training data, it will be dumped in the buffer. During the
          evaluating stage, the dumped estimators are reloaded into the
          memory sequentially to evaluate the testing data.

        By setting the ``partial_mode`` to ``True``, the partial mode is
        activated, and a local buffer will be created at the current
        directory. The partial mode is able to reduce the running memory
        cost when training the deep forest.

    Parameters
    ----------
    X : :obj:`numpy.ndarray` of shape (n_samples, n_features)
        The training data. Internally, it will be converted to
        ``np.uint8``.
    y : :obj:`numpy.ndarray` of shape (n_samples,)
        The target of input samples.
    sample_weight : :obj:`numpy.ndarray` of shape (n_samples,), default=None
        Sample weights. If ``None``, then samples are equally weighted.
"""


def deepforest_model_doc(header, item):
    """
    Decorator on obtaining documentation for deep forest models.

    Parameters
    ----------
    header: string
       Introduction to the decorated class or method.
    item : string
       Type of the docstring item.
    """

    def get_doc(item):
        """Return the selected item."""
        __doc = {
            "regressor_model": __regressor_model_doc,
            "regressor_fit": __regressor_fit_doc,
            "classifier_model": __classifier_model_doc,
            "classifier_fit": __classifier_fit_doc,
        }

        return __doc[item]

    def adddoc(cls):
        doc = [header + "\n\n"]
        doc.extend(get_doc(item))
        cls.__doc__ = "".join(doc)
        return cls

    return adddoc


class BaseCascadeForest(BaseEstimator, metaclass=ABCMeta):
    def __init__(
        self,
        n_bins=255,
        bin_subsample=2e5,
        bin_type="percentile",
        max_layers=20,
        criterion="",
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_leaf=1,
        use_predictor=False,
        predictor="forest",
        predictor_kwargs={},
        n_tolerant_rounds=2,
        delta=1e-5,
        partial_mode=False,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        self.n_bins = n_bins
        self.bin_subsample = bin_subsample
        self.bin_type = bin_type
        self.max_layers = max_layers
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.predictor_kwargs = predictor_kwargs
        self.n_tolerant_rounds = n_tolerant_rounds
        self.delta = delta
        self.partial_mode = partial_mode
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Utility variables
        self.n_layers_ = 0
        self.is_fitted_ = False

        # Internal containers
        self.layers_ = {}
        self.binners_ = {}
        self.buffer_ = _io.Buffer(partial_mode)

        # Predictor
        self.use_predictor = use_predictor
        self.predictor_name = predictor

    def __len__(self):
        return self.n_layers_

    def __getitem__(self, index):
        return self._get_layer(index)

    def _get_n_output(self, y):
        """Return the number of output inferred from the training labels."""
        if is_classifier(self):
            n_output = np.unique(y).shape[0]  # classification
            return n_output
        return 1  # this parameter are not used in regression

    def _get_layer(self, layer_idx):
        """Get the layer from the internal container according to the index."""
        if not 0 <= layer_idx < self.n_layers_:
            msg = (
                "The layer index should be in the range [0, {}], but got {}"
                " instead."
            )
            raise ValueError(msg.format(self.n_layers_ - 1, layer_idx))

        layer_key = "layer_{}".format(layer_idx)

        return self.layers_[layer_key]

    def _set_layer(self, layer_idx, layer):
        """
        Register a layer into the internal container with the given index."""
        layer_key = "layer_{}".format(layer_idx)
        if layer_key in self.layers_:
            msg = (
                "Layer with the key {} already exists in the internal"
                " container."
            )
            raise RuntimeError(msg.format(layer_key))

        self.layers_.update({layer_key: layer})

    def _get_binner(self, binner_idx):
        """Get the binner from the internal container with the given index."""
        if not 0 <= binner_idx <= self.n_layers_:
            msg = (
                "The binner index should be in the range [0, {}], but got {}"
                " instead."
            )
            raise ValueError(msg.format(self.n_layers_, binner_idx))

        binner_key = "binner_{}".format(binner_idx)

        return self.binners_[binner_key]

    def _set_binner(self, binner_idx, binner):
        """
        Register a binner into the internal container with the given index."""
        binner_key = "binner_{}".format(binner_idx)
        if binner_key in self.binners_:
            msg = (
                "Binner with the key {} already exists in the internal"
                " container."
            )
            raise RuntimeError(msg.format(binner_key))

        self.binners_.update({binner_key: binner})

    def _set_n_trees(self, layer_idx):
        """
        Set the number of decision trees for each estimator in the cascade
        layer with `layer_idx` using the pre-defined rules.
        """
        # The number of trees for each layer is fixed as `n_trees`.
        if isinstance(self.n_trees, numbers.Integral):
            if not self.n_trees > 0:
                msg = "n_trees = {} should be strictly positive."
                raise ValueError(msg.format(self.n_trees))
            return self.n_trees
        # The number of trees for the first 5 layers grows linearly with
        # `layer_idx`, while that for remaining layers is fixed to `500`.
        elif self.n_trees == "auto":
            n_trees = 100 * (layer_idx + 1)
            return n_trees if n_trees <= 500 else 500
        else:
            msg = (
                "Invalid value for n_trees. Allowed values are integers or"
                " 'auto'."
            )
            raise ValueError(msg)

    def _check_input(self, X, y=None):
        """
        Check the input data and set the attributes if X is training data."""
        is_training_data = y is not None

        if is_training_data:
            _, self.n_features_ = X.shape
            self.n_outputs_ = self._get_n_output(y)

    def _validate_params(self):
        """
        Validate parameters, those passed to the sub-modules will not be
        checked here."""
        if not self.n_outputs_ > 0:
            msg = "n_outputs = {} should be strictly positive."
            raise ValueError(msg.format(self.n_outputs_))

        if not self.max_layers > 0:
            msg = "max_layers = {} should be strictly positive."
            raise ValueError(msg.format(self.max_layers))

        if not self.n_tolerant_rounds > 0:
            msg = "n_tolerant_rounds = {} should be strictly positive."
            raise ValueError(msg.format(self.n_tolerant_rounds))

        if not self.delta >= 0:
            msg = "delta = {} should be no smaller than 0."
            raise ValueError(msg.format(self.delta))

    def _bin_data(self, binner, X, is_training_data=True):
        """
        Bin data X. If X is training data, the bin mapper is fitted first."""
        description = "training" if is_training_data else "testing"

        tic = time.time()
        if is_training_data:
            X_binned = binner.fit_transform(X)
        else:
            X_binned = binner.transform(X)
            X_binned = np.ascontiguousarray(X_binned)
        toc = time.time()
        binning_time = toc - tic

        if self.verbose > 1:
            msg = (
                "{} Binning {} data: {:.3f} MB => {:.3f} MB |"
                " Elapsed = {:.3f} s"
            )
            print(
                msg.format(
                    _utils.ctime(),
                    description,
                    X.nbytes / (1024 * 1024),
                    X_binned.nbytes / (1024 * 1024),
                    binning_time,
                )
            )

        return X_binned

    def _handle_early_stopping(self):
        """
        Remove cascade layers temporarily added, along with dumped objects on
        the local buffer if `partial_mode` is True."""
        for layer_idx in range(
            self.n_layers_ - 1, self.n_layers_ - self.n_tolerant_rounds, -1
        ):
            self.layers_.pop("layer_{}".format(layer_idx))
            self.binners_.pop("binner_{}".format(layer_idx))

            if self.partial_mode:
                self.buffer_.del_estimator(layer_idx)

        # The last layer temporarily added only requires dumped estimators on
        # the local buffer to be removed.
        if self.partial_mode:
            self.buffer_.del_estimator(self.n_layers_)

        self.n_layers_ -= self.n_tolerant_rounds - 1

        if self.verbose > 0:
            msg = "{} The optimal number of layers: {}"
            print(msg.format(_utils.ctime(), self.n_layers_))

    def _if_improved(self, new_pivot, pivot, delta, is_classifier):
        """
        Return true if new vlidation result is better than previous"""
        if is_classifier:
            return new_pivot >= pivot + delta
        return new_pivot <= pivot - delta

    @abstractmethod
    def _repr_performance(self, pivot):
        """Format the printting information on training performance."""

    @abstractmethod
    def predict(self, X):
        """
        Predict class labels or regression values for X.
        For classification, the predicted class for each sample in X is
        returned. For regression, the predicted value based on X is returned.
        """

    @property
    def n_aug_features_(self):
        if is_classifier(self):
            return 2 * self.n_estimators * self.n_outputs_
        return 2 * self.n_estimators

    # flake8: noqa: E501
    def fit(self, X, y, sample_weight=None):

        self._check_input(X, y)
        self._validate_params()
        n_counter = 0  # a counter controlling the early stopping

        binner_ = Binner(
            n_bins=self.n_bins,
            bin_subsample=self.bin_subsample,
            bin_type=self.bin_type,
            random_state=self.random_state,
        )

        # Bin the training data
        X_train_ = self._bin_data(binner_, X, is_training_data=True)
        X_train_ = self.buffer_.cache_data(0, X_train_, is_training_data=True)

        # =====================================================================
        # Training Stage
        # =====================================================================

        if self.verbose > 0:
            print("{} Start to fit the model:".format(_utils.ctime()))

        # Build the first cascade layer
        layer_ = Layer(
            0,
            self.n_outputs_,
            self.criterion,
            self.n_estimators,
            self._set_n_trees(0),
            self.max_depth,
            self.min_samples_leaf,
            self.partial_mode,
            self.buffer_,
            self.n_jobs,
            self.random_state,
            self.verbose,
            is_classifier(self),
        )

        if self.verbose > 0:
            print("{} Fitting cascade layer = {:<2}".format(_utils.ctime(), 0))

        tic = time.time()
        X_aug_train_ = layer_.fit_transform(
            X_train_, y, sample_weight=sample_weight
        )
        toc = time.time()
        training_time = toc - tic

        # Set the reference performance
        pivot = layer_.val_acc_

        if self.verbose > 0:
            msg = "{} layer = {:<2} | {} | Elapsed = {:.3f} s"
            print(
                msg.format(
                    _utils.ctime(),
                    0,
                    self._repr_performance(pivot),
                    training_time,
                )
            )

        # Copy the snapshot of `X_aug_train_` for training the predictor.
        if self.use_predictor:
            snapshot_X_aug_train_ = np.copy(X_aug_train_)

        # Add the first cascade layer, binner
        self._set_layer(0, layer_)
        self._set_binner(0, binner_)
        self.n_layers_ += 1

        # Pre-allocate the global array on storing training data
        X_middle_train_ = _utils.init_array(X_train_, self.n_aug_features_)

        # ====================================================================
        # Main loop on the training stage
        # ====================================================================

        while self.n_layers_ < self.max_layers:

            # Set the binner
            binner_ = Binner(
                n_bins=self.n_bins,
                bin_subsample=self.bin_subsample,
                bin_type=self.bin_type,
                random_state=self.random_state,
            )

            X_binned_aug_train_ = self._bin_data(
                binner_, X_aug_train_, is_training_data=True
            )

            X_middle_train_ = _utils.merge_array(
                X_middle_train_, X_binned_aug_train_, self.n_features_
            )

            # Build a cascade layer
            layer_idx = self.n_layers_
            layer_ = Layer(
                layer_idx,
                self.n_outputs_,
                self.criterion,
                self.n_estimators,
                self._set_n_trees(layer_idx),
                self.max_depth,
                self.min_samples_leaf,
                self.partial_mode,
                self.buffer_,
                self.n_jobs,
                self.random_state,
                self.verbose,
                is_classifier(self),
            )

            X_middle_train_ = self.buffer_.cache_data(
                layer_idx, X_middle_train_, is_training_data=True
            )

            if self.verbose > 0:
                msg = "{} Fitting cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            tic = time.time()
            X_aug_train_ = layer_.fit_transform(
                X_middle_train_, y, sample_weight=sample_weight
            )
            toc = time.time()
            training_time = toc - tic

            new_pivot = layer_.val_acc_

            if self.verbose > 0:
                msg = "{} layer = {:<2} | {} | Elapsed = {:.3f} s"
                print(
                    msg.format(
                        _utils.ctime(),
                        layer_idx,
                        self._repr_performance(new_pivot),
                        training_time,
                    )
                )

            # Check on early stopping: If the performance of the fitted
            # cascade layer does not improved by `delta` compared to the best
            # performance achieved so far for `n_tolerant_rounds`, the
            # training stage will terminate before reaching the maximum number
            # of layers.

            if self._if_improved(
                new_pivot, pivot, self.delta, is_classifier(self)
            ):

                # Update the cascade layer
                self._set_layer(layer_idx, layer_)
                self._set_binner(layer_idx, binner_)
                self.n_layers_ += 1

                # Performance calibration
                n_counter = 0
                pivot = new_pivot

                if self.use_predictor:
                    snapshot_X_aug_train_ = np.copy(X_aug_train_)
            else:
                n_counter += 1

                if self.verbose > 0:
                    msg = "{} Early stopping counter: {} out of {}"
                    print(
                        msg.format(
                            _utils.ctime(), n_counter, self.n_tolerant_rounds
                        )
                    )

                # Activate early stopping if reaching `n_tolerant_rounds`
                if n_counter == self.n_tolerant_rounds:

                    if self.verbose > 0:
                        msg = "{} Handling early stopping"
                        print(msg.format(_utils.ctime()))

                    self._handle_early_stopping()
                    break

                # Add the fitted layer, and binner temporarily
                self._set_layer(layer_idx, layer_)
                self._set_binner(layer_idx, binner_)
                self.n_layers_ += 1

        if self.n_layers_ == self.max_layers and self.verbose > 0:
            msg = "{} Reaching the maximum number of layers: {}"
            print(msg.format(_utils.ctime(), self.max_layers))

        # Build the predictor if `self.use_predictor` is True
        if self.use_predictor:
            if is_classifier(self):
                self.predictor_ = _build_classifier_predictor(
                    self.predictor_name,
                    self.criterion,
                    self.n_trees,
                    self.n_outputs_,
                    self.max_depth,
                    self.min_samples_leaf,
                    self.n_jobs,
                    self.random_state,
                    self.predictor_kwargs,
                )
            else:
                self.predictor_ = _build_regressor_predictor(
                    self.predictor_name,
                    self.criterion,
                    self.n_trees,
                    self.n_outputs_,
                    self.max_depth,
                    self.min_samples_leaf,
                    self.n_jobs,
                    self.random_state,
                    self.predictor_kwargs,
                )

            binner_ = Binner(
                n_bins=self.n_bins,
                bin_subsample=self.bin_subsample,
                bin_type=self.bin_type,
                random_state=self.random_state,
            )

            X_binned_aug_train_ = self._bin_data(
                binner_, snapshot_X_aug_train_, is_training_data=True
            )

            X_middle_train_ = _utils.merge_array(
                X_middle_train_, X_binned_aug_train_, self.n_features_
            )

            if self.verbose > 0:
                msg = "{} Fitting the concatenated predictor: {}"
                print(msg.format(_utils.ctime(), self.predictor_name))

            tic = time.time()
            self.predictor_.fit(
                X_middle_train_, y, sample_weight=sample_weight
            )
            toc = time.time()

            if self.verbose > 0:
                msg = "{} Finish building the predictor | Elapsed = {:.3f} s"
                print(msg.format(_utils.ctime(), toc - tic))

            self._set_binner(self.n_layers_, binner_)
            self.predictor_ = self.buffer_.cache_predictor(self.predictor_)

        self.is_fitted_ = True

        return self

    def get_forest(self, layer_idx, est_idx, forest_type):
        """
        Get the `est_idx`-th forest estimator from the `layer_idx`-th
        cascade layer in the model.

        Parameters
        ----------
        layer_idx : :obj:`int`
            The index of the cascade layer, should be in the range
            ``[0, self.n_layers_-1]``.
        est_idx : :obj:`int`
            The index of the forest estimator, should be in the range
            ``[0, self.n_estimators]``.
        forest_type : :obj:`{"rf", "erf"}`
            Specify the forest type.

            - If ``rf``, return the random forest.
            - If ``erf``, return the extremely random forest.

        Returns
        -------
        estimator : The forest estimator with the given index.
        """
        if not self.is_fitted_:
            raise AttributeError("Please fit the model first.")

        # Check the given index
        if not 0 <= layer_idx < self.n_layers_:
            msg = (
                "`layer_idx` should be in the range [0, {}), but got"
                " {} instead."
            )
            raise ValueError(msg.format(self.n_layers_, layer_idx))

        if not 0 <= est_idx < self.n_estimators:
            msg = (
                "`est_idx` should be in the range [0, {}), but got"
                " {} instead."
            )
            raise ValueError(msg.format(self.n_estimators, est_idx))

        if forest_type not in ("rf", "erf"):
            msg = (
                "`forest_type` should be one of {{rf, erf}},"
                " but got {} instead."
            )
            raise ValueError(msg.format(forest_type))

        layer = self._get_layer(layer_idx)
        est_key = "{}-{}-{}".format(layer_idx, est_idx, forest_type)
        estimator = layer.estimators_[est_key]

        # Load the model if in partial mode
        if self.partial_mode:
            estimator = self.buffer_.load_estimator(estimator)

        return estimator.estimator_

    def save(self, dirname="model"):
        """
        Save the model to the specified directory.

        Parameters
        ----------
        dirname : :obj:`str`, default="model"
            The name of the output directory.


        .. warning::
            Other methods on model serialization such as :mod:`pickle` or
            :mod:`joblib` are not recommended, especially when ``partial_mode``
            is set to ``True``.
        """
        # Create the output directory
        _io.model_mkdir(dirname)

        # Save each object sequentially
        d = {}
        d["n_estimators"] = self.n_estimators
        d["criterion"] = self.criterion
        d["n_layers"] = self.n_layers_
        d["n_features"] = self.n_features_
        d["n_outputs"] = self.n_outputs_
        d["partial_mode"] = self.partial_mode
        d["buffer"] = self.buffer_
        d["verbose"] = self.verbose
        d["use_predictor"] = self.use_predictor

        if self.use_predictor:
            d["predictor_name"] = self.predictor_name

        # Save label encoder if labels are encoded.
        if hasattr(self, "labels_are_encoded"):
            d["labels_are_encoded"] = self.labels_are_encoded
            d["label_encoder"] = self.label_encoder_

        _io.model_saveobj(dirname, "param", d)
        _io.model_saveobj(dirname, "binner", self.binners_)
        _io.model_saveobj(dirname, "layer", self.layers_, self.partial_mode)

        if self.use_predictor:
            _io.model_saveobj(
                dirname, "predictor", self.predictor_, self.partial_mode
            )

    def load(self, dirname):
        """
        Load the model from the specified directory.

        Parameters
        ----------
        dirname : :obj:`str`
            The name of the input directory.


        .. note::
            The dumped model after calling :meth:`load_model` is not exactly
            the same as the model before saving, because many objects
            irrelevant to model inference will not be saved.
        """
        d = _io.model_loadobj(dirname, "param")

        # Set parameter
        self.n_estimators = d["n_estimators"]
        self.n_layers_ = d["n_layers"]
        self.n_features_ = d["n_features"]
        self.n_outputs_ = d["n_outputs"]
        self.partial_mode = d["partial_mode"]
        self.verbose = d["verbose"]
        self.use_predictor = d["use_predictor"]

        # Load label encoder if labels are encoded.
        if "labels_are_encoded" in d:
            self.labels_are_encoded = d["labels_are_encoded"]
            self.label_encoder_ = d["label_encoder"]

        # Load internal containers
        self.binners_ = _io.model_loadobj(dirname, "binner")
        self.layers_ = _io.model_loadobj(dirname, "layer", d)
        if self.use_predictor:
            self.predictor_ = _io.model_loadobj(dirname, "predictor", d)

        # Some checks after loading
        if len(self.layers_) != self.n_layers_:
            msg = (
                "The size of the loaded dictionary of layers {} does not"
                " match n_layers_ {}."
            )
            raise RuntimeError(msg.format(len(self.layers_), self.n_layers_))

        self.is_fitted_ = True

    def clean(self):
        """
        Clean the buffer created by the model if ``partial_mode`` is ``True``.
        """
        if self.partial_mode:
            self.buffer_.close()


@deepforest_model_doc(
    """Implementation of the deep forest for classification.""",
    "classifier_model",
)
class CascadeForestClassifier(BaseCascadeForest, ClassifierMixin):
    def __init__(
        self,
        n_bins=255,
        bin_subsample=2e5,
        bin_type="percentile",
        max_layers=20,
        criterion="gini",
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_leaf=1,
        use_predictor=False,
        predictor="forest",
        predictor_kwargs={},
        n_tolerant_rounds=2,
        delta=1e-5,
        partial_mode=False,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        super().__init__(
            n_bins=n_bins,
            bin_subsample=bin_subsample,
            bin_type=bin_type,
            max_layers=max_layers,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            use_predictor=use_predictor,
            predictor=predictor,
            predictor_kwargs=predictor_kwargs,
            n_tolerant_rounds=n_tolerant_rounds,
            delta=delta,
            partial_mode=partial_mode,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        # Used to deal with classification labels
        self.labels_are_encoded = False
        self.type_of_target_ = None
        self.label_encoder_ = None

    def _encode_class_labels(self, y):
        """
        Fit the internal label encoder and return encoded labels.
        """
        self.type_of_target_ = type_of_target(y)
        if self.type_of_target_ in ("binary", "multiclass"):
            self.labels_are_encoded = True
            self.label_encoder_ = LabelEncoder()
            encoded_y = self.label_encoder_.fit_transform(y)
        else:
            msg = (
                "CascadeForestClassifier is used for binary and multiclass"
                " classification, wheras the training labels seem not to"
                " be any one of them."
            )
            raise ValueError(msg)

        return encoded_y

    def _decode_class_labels(self, y):
        """
        Transform the predicted labels back to original encoding.
        """
        if self.labels_are_encoded:
            decoded_y = self.label_encoder_.inverse_transform(y)
        else:
            decoded_y = y

        return decoded_y

    def _repr_performance(self, pivot):
        msg = "Val Acc = {:.3f} %"
        return msg.format(pivot * 100)

    @deepforest_model_doc(
        """Build a deep forest using the training data.""", "classifier_fit"
    )
    def fit(self, X, y, sample_weight=None):

        # Check the input for classification
        y = self._encode_class_labels(y)

        super().fit(X, y, sample_weight)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : :obj:`numpy.ndarray` of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``np.uint8``.

        Returns
        -------
        proba : :obj:`numpy.ndarray` of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        if not self.is_fitted_:
            raise AttributeError("Please fit the model first.")
        self._check_input(X)

        if self.verbose > 0:
            print("{} Start to evalute the model:".format(_utils.ctime()))

        binner_ = self._get_binner(0)
        X_test = self._bin_data(binner_, X, is_training_data=False)
        X_middle_test_ = _utils.init_array(X_test, self.n_aug_features_)

        for layer_idx in range(self.n_layers_):
            layer = self._get_layer(layer_idx)

            if self.verbose > 0:
                msg = "{} Evaluating cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            if layer_idx == 0:
                X_aug_test_ = layer.transform(X_test, is_classifier(self))
            elif layer_idx < self.n_layers_ - 1:
                binner_ = self._get_binner(layer_idx)
                X_aug_test_ = self._bin_data(
                    binner_, X_aug_test_, is_training_data=False
                )
                X_middle_test_ = _utils.merge_array(
                    X_middle_test_, X_aug_test_, self.n_features_
                )
                X_aug_test_ = layer.transform(
                    X_middle_test_, is_classifier(self)
                )
            else:
                binner_ = self._get_binner(layer_idx)
                X_aug_test_ = self._bin_data(
                    binner_, X_aug_test_, is_training_data=False
                )
                X_middle_test_ = _utils.merge_array(
                    X_middle_test_, X_aug_test_, self.n_features_
                )

                # Skip calling the `transform` if not using the predictor
                if self.use_predictor:
                    X_aug_test_ = layer.transform(
                        X_middle_test_, is_classifier(self)
                    )

        if self.use_predictor:

            if self.verbose > 0:
                print("{} Evaluating the predictor".format(_utils.ctime()))

            binner_ = self._get_binner(self.n_layers_)
            X_aug_test_ = self._bin_data(
                binner_, X_aug_test_, is_training_data=False
            )
            X_middle_test_ = _utils.merge_array(
                X_middle_test_, X_aug_test_, self.n_features_
            )

            predictor = self.buffer_.load_predictor(self.predictor_)
            proba = predictor.predict_proba(X_middle_test_)
        else:
            proba = layer.predict_full(X_middle_test_, is_classifier(self))
            proba = _utils.merge_proba(proba, self.n_outputs_)

        return proba

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : :obj:`numpy.ndarray` of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``np.uint8``.

        Returns
        -------
        y : :obj:`numpy.ndarray` of shape (n_samples,)
            The predicted classes.
        """
        proba = self.predict_proba(X)
        y = self._decode_class_labels(np.argmax(proba, axis=1))
        return y


@deepforest_model_doc(
    """Implementation of the deep forest for regression.""", "regressor_model"
)
class CascadeForestRegressor(BaseCascadeForest, RegressorMixin):
    def __init__(
        self,
        n_bins=255,
        bin_subsample=2e5,
        bin_type="percentile",
        max_layers=20,
        criterion="mse",
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_leaf=1,
        use_predictor=False,
        predictor="forest",
        predictor_kwargs={},
        n_tolerant_rounds=2,
        delta=1e-5,
        partial_mode=False,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        super().__init__(
            n_bins=n_bins,
            bin_subsample=bin_subsample,
            bin_type=bin_type,
            max_layers=max_layers,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            use_predictor=use_predictor,
            predictor=predictor,
            predictor_kwargs=predictor_kwargs,
            n_tolerant_rounds=n_tolerant_rounds,
            delta=delta,
            partial_mode=partial_mode,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _repr_performance(self, pivot):
        msg = "Val Acc = {:.3f}"
        return msg.format(pivot)

    @deepforest_model_doc(
        """Build a deep forest using the training data.""", "regressor_fit"
    )
    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)

    def predict(self, X):
        """
        Predict regression target for X.

        Parameters
        ----------
        X : :obj:`numpy.ndarray` of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``np.uint8``.

        Returns
        -------
        y : :obj:`numpy.ndarray` of shape (n_samples,)
            The predicted values.
        """
        if not self.is_fitted_:
            raise AttributeError("Please fit the model first.")
        self._check_input(X)

        if self.verbose > 0:
            print("{} Start to evalute the model:".format(_utils.ctime()))

        binner_ = self._get_binner(0)
        X_test = self._bin_data(binner_, X, is_training_data=False)
        X_middle_test_ = _utils.init_array(X_test, self.n_aug_features_)

        for layer_idx in range(self.n_layers_):
            layer = self._get_layer(layer_idx)

            if self.verbose > 0:
                msg = "{} Evaluating cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            if layer_idx == 0:
                X_aug_test_ = layer.transform(X_test, is_classifier(self))
            elif layer_idx < self.n_layers_ - 1:
                binner_ = self._get_binner(layer_idx)
                X_aug_test_ = self._bin_data(
                    binner_, X_aug_test_, is_training_data=False
                )
                X_middle_test_ = _utils.merge_array(
                    X_middle_test_, X_aug_test_, self.n_features_
                )
                X_aug_test_ = layer.transform(
                    X_middle_test_, is_classifier(self)
                )
            else:
                binner_ = self._get_binner(layer_idx)
                X_aug_test_ = self._bin_data(
                    binner_, X_aug_test_, is_training_data=False
                )
                X_middle_test_ = _utils.merge_array(
                    X_middle_test_, X_aug_test_, self.n_features_
                )

                # Skip calling the `transform` if not using the predictor
                if self.use_predictor:
                    X_aug_test_ = layer.transform(
                        X_middle_test_, is_classifier(self)
                    )

        if self.use_predictor:

            if self.verbose > 0:
                print("{} Evaluating the predictor".format(_utils.ctime()))

            binner_ = self._get_binner(self.n_layers_)
            X_aug_test_ = self._bin_data(
                binner_, X_aug_test_, is_training_data=False
            )
            X_middle_test_ = _utils.merge_array(
                X_middle_test_, X_aug_test_, self.n_features_
            )

            predictor = self.buffer_.load_predictor(self.predictor_)
            _y = predictor.predict(X_middle_test_)
        else:
            _y = layer.predict_full(X_middle_test_, is_classifier(self))
            _y = _y.sum(axis=1) / _y.shape[1]
        return _y
