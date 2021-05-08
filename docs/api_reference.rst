API Reference
=============

Below is the class and function reference for :mod:`deepforest`. Notice that the package is still under active development, and some features may not be stable yet.

.. currentmodule:: deepforest.CascadeForestClassifier

CascadeForestClassifier
-----------------------

.. autosummary::

    fit
    predict_proba
    predict
    clean
    get_estimator
    get_layer_feature_importances
    load
    save
    set_estimator
    set_predictor

.. autoclass:: deepforest.CascadeForestClassifier
    :members:
    :inherited-members:
    :show-inheritance:
    :no-undoc-members:
    :exclude-members: set_params, get_params, score
    :member-order: bysource

.. currentmodule:: deepforest.CascadeForestRegressor

CascadeForestRegressor
-----------------------

.. autosummary::

    fit
    predict
    clean
    get_estimator
    get_layer_feature_importances
    load
    save
    set_estimator
    set_predictor

.. autoclass:: deepforest.CascadeForestRegressor
    :members:
    :inherited-members:
    :show-inheritance:
    :no-undoc-members:
    :exclude-members: set_params, get_params, score
    :member-order: bysource
