Use Customized Estimators
=========================

The version v0.1.4 of :mod:`deepforest` has added the support on:

- using customized base estimators in cascade layers of deep forest
- using the customized predictor concatenated to the deep forest

The page gives a detailed introduction on how to use this new feature.

Instantiate the deep forest model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To begin with, you need to instantiate a deep forest model. Notice that some parameters specified here will be overridden by downstream steps. For example, if the parameter :obj:`use_predictor` is set to ``False`` here, whereas :meth:`set_predictor` is called latter, then the internal attribute :obj:`use_predictor` will be altered to ``True``.

.. code-block:: python

    from deepforest import CascadeForestClassifier
    model = CascadeForestClassifier()

Instantiate your estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to use customized estimators in the cascade layer of deep forest, the next step is to instantiate the estimators and encapsulate them into a Python list:

.. code-block:: python

    n_estimators = 4  # the number of base estimators per cascade layer
    estimators = [your_estimator(random_state=i) for i in range(n_estimators)]

.. tip::

    You need to make sure that instantiated estimators in the list are with different random seeds if seeds are manually specified. Otherwise, they will have the same behavior on the dataset and make cascade layers less effective.

For the customized predictor, you only need to instantiate it, and there is no extra step:

.. code-block:: python

    predictor = your_predictor()

Deep forest will conduct internal checks to make sure that :obj:`estimators` and :obj:`predictor` are valid for training and evaluating. To pass the internal checks, the class of your customized estimators or predictor should at least implement methods listed below:

* :meth:`fit` for training
* **[Classification]** :meth:`predict_proba` for evaluating
* **[Regression]** :meth:`predict` for evaluating

The name of these methods follow the convention in scikit-learn, and they are already implemented in a lot of packages offering scikit-learn APIs (e.g., `XGBoost <https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn>`__, `LightGBM <https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api>`__, `CatBoost <https://catboost.ai/docs/concepts/python-quickstart.html>`__). Otherwise, you have to implement a wrapper on your customized estimators to make these methods callable.

Call :meth:`set_estimator` and :meth:`set_predictor`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core step is to call :meth:`set_estimator` and :meth:`set_predictor` to override estimators used by default:

.. code-block:: python

    # Customized base estimators
    model.set_estimator(estimators)

    # Customized predictor
    model.set_predictor(predictor)

:meth:`set_estimator` has another parameter :obj:`n_splits`, which determines the number of folds of the internal cross-validation strategy. Its value should be at least ``2``, and the default value is ``5``. Generally speaking, a larger :obj:`n_splits` leads to better generalization performance. If you are confused about the effect of cross-validation here, please refer to `the original paper <https://arxiv.org/pdf/1702.08835.pdf>`__ for details on how deep forest adopts the cross-validation strategy to build cascade layers.

Train and Evaluate
~~~~~~~~~~~~~~~~~~

Remaining steps follow the original workflow of deep forest.

.. code-block:: python

    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

.. note::

    When using customized estimators via :meth:`set_estimator`, deep forest adopts the cross-validation strategy to grow cascade layers. Suppose that :obj:`n_splits` is set to ``5`` when calling :meth:`set_estimator`, each estimator will be repeatedly trained over five times to get full augmented features from a cascade layer. As a result, you may experience a drastic increase in running time and memory.
