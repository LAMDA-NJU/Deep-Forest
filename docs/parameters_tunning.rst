Parameters Tunning
==================

This page contains parameters tuning guides for deep forest.

Better Accuracy
---------------

Increase model complexity
*************************

An intuitive way of improving the performance of deep forest is to increase its model complexity, below are some important parameters controlling the model complexity:

- ``n_estimators``: Specify the number of estimators in each cascade layer.
- ``n_trees``: Specify the number of trees in each estimator.
- ``max_layers``: Specify the maximum number of cascade layers.

Using a large value of parameters above, the performance of deep forest may improve on complex datasets that require a larger model to perform well.
 
Add the predictor
*****************

In addition to increasing the model complexity, you can borrow the power of random forest or gradient boosting decision tree (GBDT), which can be helpful depending on the dataset:

- ``use_predictor``: Decide whether to use the predictor concatenated to the deep forest.
- ``predictor``: Specify the type of the predictor, should be one of ``"forest"``, ``"xgboost"``, ``"lightgbm"``.

Please make sure that XGBoost or LightGBM are installed if you are going to use them as the predictor.

.. tip::    
    A useful rule on deciding whether to add the predictor is to compare the performance of deep forest with a standalone predictor produced from the training data. If the predictor consistently outperforms deep forest, then the performance of deep forest is expected to improve via adding the predictor. In this case, the augmented features produced from deep forest also facilitate training the predictor. 

Faster Speed
------------

Parallelization
***************

Using parallelization is highly recommended because deep forest is naturally suited to it.

- ``n_jobs``: Specify the number of workers used. Setting its value to an integer larger than 1 enables the parallelization. Setting its value to ``-1`` means all processors are used.

Fewer Splits
************

- ``n_bins``: Specify the number of feature discrete bins. A smaller value means fewer splitting cut-offs will be considered, should be an integer in the range [2, 255].
- ``bin_type``: Specify the binning type. Setting its value to ``"interval"`` enables less splitting cut-offs to be considered on dense intervals where the feature values accumulate.

Decrease model complexity
*************************

Setting parameters below to a smaller value decreases the model complexity of deep forest, and may lead to a faster speed on training and evaluating.

- ``max_depth``: Specify the maximum depth of tree. ``None`` indicates no constraint.
- ``min_samples_leaf``: Specify the minimum number of samples required to be at a leaf node. The smallest value is ``1``.
- ``n_estimators``: Specify the number of estimators in each cascade layer.
- ``n_trees``: Specify the number of trees in each estimator.
- ``n_tolerant_rounds``: Specify the number of tolerant rounds when handling early stopping. The smallest value is ``1``.

.. warning::
    Since deep forest automatically determines the model complexity according to the validation performance on the training data, setting parameters above to a smaller value may lead to a deep forest model with more cascade layers.

Lower Memory Usage
-------------------

Partial Mode
************

- ``partial_mode``: Decide whether to train and evaluate the model in partial mode. If set to ``True``, the model will actively dump fitted estimators in a local buffer. As a result, **the memory usage of deep forest no longer increases linearly with the number of fitted cascade layers**.

In addition, decreasing the model complexity also pulls down the memory usage.
