Deep Forest (DF) 21
===================

|github|_ |readthedocs|_ |codecov|_ |python|_ |pypi|_ |style|_

.. |github| image:: https://github.com/LAMDA-NJU/Deep-Forest/workflows/DeepForest-CI/badge.svg
.. _github: https://github.com/LAMDA-NJU/Deep-Forest/actions

.. |readthedocs| image:: https://readthedocs.org/projects/deep-forest/badge/?version=latest
.. _readthedocs: https://deep-forest.readthedocs.io

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

DF21 offers an effective & powerful option to the tree-based machine learning algorithms such as Random Forest or GBDT.

For a quick start, please refer to `How to Get Started <https://deep-forest.readthedocs.io/en/latest/how_to_get_started.html>`__. For a detailed guidance on parameter tunning, please refer to `Parameters Tunning <https://deep-forest.readthedocs.io/en/latest/parameters_tunning.html>`__.

DF21 is optimized for what a tree-based ensemble excels at (i.e., tabular data), if you want to use the multi-grained scanning part to better handle structured data like images, please refer to the `origin implementation <https://github.com/kingfengji/gcForest>`__ for details.

Installation
------------

DF21 can be installed using pip via `PyPI <https://pypi.org/project/deep-forest/>`__  which is the package installer for Python. You can use pip to install packages from the Python Package Index and other indexes. Refer `this <https://pypi.org/project/pip/>`__ for the documentation of pip. Use this command to download DF21 :

.. code-block:: bash

    pip install deep-forest

Quickstart
----------

Classification
**************

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

Regression
**********

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

* `Documentation <https://deep-forest.readthedocs.io/>`__
* Deep Forest: `[Conference] <https://www.ijcai.org/proceedings/2017/0497.pdf>`__ | `[Journal] <https://academic.oup.com/nsr/article-pdf/6/1/74/30336169/nwy108.pdf>`__
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
        title = {{Deep Forest:} Towards an alternative to deep neural networks},
        author = {Zhi-Hua Zhou and Ji Feng},
        booktitle = {IJCAI},
        pages = {3553--3559},
        year = {2017}}

Thanks to all our contributors
------------------------------

|contributors|

.. |contributors| image:: https://contributors-img.web.app/image?repo=LAMDA-NJU/Deep-Forest
.. _contributors: https://github.com/LAMDA-NJU/Deep-Forest/graphs/contributors
