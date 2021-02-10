DF21 Documentation
==================

**DF21** is an implementation of `Deep Forest <https://arxiv.org/pdf/1702.08835.pdf>`__ 2021.2.1. It is designed to have the following advantages:

- **Powerful**: Better accuracy than existing tree-based ensemble methods.
- **Easy to Use**: Less efforts on tunning parameters.
- **Efficient**: Fast training speed and high efficiency.
- **Scalable**: Capable of handling large-scale data.

Whenever one used tree-based machine learning approaches such as Random Forest or GBDT, DF21 may offer a new powerful option. This package is actively being developed, and any help would be welcomed. Please check the homepage on `Gitee <https://gitee.com/lamda-nju/deep-forest>`__ or `Github <https://github.com/LAMDA-NJU/Deep-Forest>`__ for details.

Guidepost
---------
* For a quick start, please refer to `How to Get Started <./how_to_get_started.html>`__.
* For a guidance on tunning parameters for DF21, please refer to `Parameters Tunning <./parameters_tunning.html>`__.
* For a comparison between DF21 and other tree-based ensemble methods, please refer to `Experiments <./experiments.html>`__.

Installation
------------

The package is available via `PyPI <https://pypi.org/project/deep-forest/>`__ using:

.. code-block:: bash

    $ pip install deep-forest

Quickstart
----------

.. code-block:: python

    from deepforest import CascadeForestClassifier

    # Load utils
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load data
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    model = CascadeForestClassifier(random_state=1)

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    print("Testing Accuracy: {:.3f} %".format(acc))
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
    print("\nTesting MSE: {:.3f}".format(acc))
    >>> Testing MSE: 8.068

Resources
---------

* Deep Forest: `[Paper] <https://arxiv.org/pdf/1702.08835.pdf>`__
* Keynote at AISTATS 2019: `[Slides] <https://aistats.org/aistats2019/0-AISTATS2019-slides-zhi-hua_zhou.pdf>`__
* Source Code: `[Gitee] <https://gitee.com/lamda-nju/deep-forest>`__ | `[GitHub] <https://github.com/LAMDA-NJU/Deep-Forest>`__

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

.. toctree::
   :maxdepth: 1
   :caption: For Users

   How to Get Started <how_to_get_started>
   Installation Guide <installation_guide>
   API Reference <api_reference>
   Parameters Tunning <parameters_tunning>
   Experiments <experiments>

.. toctree::
   :maxdepth: 1
   :caption: For Developers

   Changelog <changelog>

.. toctree::
   :maxdepth: 1
   :caption: About

   About Us <http://www.lamda.nju.edu.cn/MainPage.ashx>
   Related Software <related_software>

Acknowledgement
---------------

The lead developer and maintainer of DF21 is Mr. `Yi-Xuan Xu <https://github.com/xuyxu>`__. Before the release, it has been used internally in the LAMDA Group, Nanjing University, China.
