Deep Forest (DF) 21
===================

|github|_ |readthedocs|_ |codecov|_ |python|_ |pypi|_ |style|_

.. |github| image:: https://github.com/LAMDA-NJU/Deep-Forest/workflows/DeepForest-CI/badge.svg
.. _github: https://github.com/LAMDA-NJU/Deep-Forest/actions

.. |readthedocs| image:: https://readthedocs.org/projects/deep-forest/badge/?version=latest
.. _readthedocs: https://deep-forest.readthedocs.io/en/latest/

.. |codecov| image:: https://codecov.io/gh/LAMDA-NJU/Deep-Forest/branch/master/graph/badge.svg?token=5BVXOT8RPO
.. _codecov: https://codecov.io/gh/LAMDA-NJU/Deep-Forest
    
.. |python| image:: https://img.shields.io/pypi/pyversions/deep-forest
.. _python: https://pypi.org/project/deep-forest/

.. |pypi| image:: https://img.shields.io/pypi/v/deep-forest?color=blue
.. _pypi: https://pypi.org/project/deep-forest/

.. |style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _style: https://github.com/psf/black

**DF21** is an implementation of `Deep Forest <https://arxiv.org/pdf/1702.08835.pdf>`__ 2021.2.1. It is designed to have the following advantages:

- **Powerful**: Better accuracy than existing tree-based ensemble methods.
- **Easy to Use**: Less efforts on tunning parameters.
- **Efficient**: Fast training speed and high efficiency.
- **Scalable**: Capable of handling large-scale data.

Whenever one used tree-based machine learning approaches such as Random Forest or GBDT, DF21 may offer a new powerful option.

For a quick start, please refer to `How to Get Started <https://deep-forest.readthedocs.io/en/latest/how_to_get_started.html>`__. For a detailed guidance on parameter tunning, please refer to `Parameters Tunning <https://deep-forest.readthedocs.io/en/latest/parameters_tunning.html>`__.

Installation
------------

The package is available via PyPI using:

.. code-block:: bash

    pip install deep-forest

Quickstart
----------

.. code-block:: python

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    from deepforest import CascadeForestClassifier

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model = CascadeForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print("\nTesting Accuracy: {:.3f} %".format(acc))
    >>> Testing Accuracy: 98.667 %

.. code-block:: python

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    from deepforest import CascadeForestRegressor

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model = CascadeForestRegressor(random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("\nTesting MSE: {:.3f}".format(mse))
    >>> Testing MSE: 8.068

Resources
---------

* `Documentation <https://deep-forest.readthedocs.io/en/latest/>`__
* Deep Forest: `[Paper] <https://arxiv.org/pdf/1702.08835.pdf>`__
* Keynote at AISTATS 2019: `[Slides] <https://aistats.org/aistats2019/0-AISTATS2019-slides-zhi-hua_zhou.pdf>`__

Reference
---------

.. code-block:: latex

    @article{zhou2019deep,
        title={Deep forest},
        author={Zhi-Hua Zhou and Ji Feng},
        journal={National Science Review},
        volume={6},
        number={1},
        pages={74--86},
        year={2019}}

    @inproceedings{zhou2017deep,
        Author = {Zhi-Hua Zhou and Ji Feng},
        Booktitle = {IJCAI},
        Pages = {3553-3559},
        Title = {{Deep Forest:} Towards an alternative to deep neural networks},
        Year = {2017}}

Acknowledgement
---------------

The lead developer and maintainer of DF21 is Mr. `Yi-Xuan Xu <https://github.com/xuyxu>`__. Before the release, it has been used internally in the LAMDA Group, Nanjing University, China.
