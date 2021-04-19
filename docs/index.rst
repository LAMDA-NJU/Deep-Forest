DF21 Documentation
==================

**DF21** is an implementation of `Deep Forest <https://arxiv.org/pdf/1702.08835.pdf>`__ 2021.2.1. It is designed to have the following advantages:

- **Powerful**: Better accuracy than existing tree-based ensemble methods.
- **Easy to Use**: Less efforts on tunning parameters.
- **Efficient**: Fast training speed and high efficiency.
- **Scalable**: Capable of handling large-scale data.

DF21 offers an effective & powerful option to the tree-based machine learning algorithms such as Random Forest or GBDT. This package is actively being developed, and any help would be welcomed. Please check the homepage on `Gitee <https://gitee.com/lamda-nju/deep-forest>`__ or `Github <https://github.com/LAMDA-NJU/Deep-Forest>`__ for details.

Guidepost
---------
* For a quick start, please refer to `How to Get Started <./how_to_get_started.html>`__.
* For a guidance on tunning parameters for DF21, please refer to `Parameters Tunning <./parameters_tunning.html>`__.
* For a comparison between DF21 and other tree-based ensemble methods, please refer to `Experiments <./experiments.html>`__.

Installation
------------

DF21 can be installed using pip via `PyPI <https://pypi.org/project/deep-forest/>`__  which is the package installer for Python. You can use pip to install packages from the Python Package Index and other indexes. Refer `this <https://pypi.org/project/pip/>`__ for the documentation of pip. Use this command to download DF21 :

.. code-block:: bash

    $ pip install deep-forest

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

* Deep Forest: `[Paper] <https://arxiv.org/pdf/1702.08835.pdf>`__
* Keynote at AISTATS 2019: `[Slides] <https://aistats.org/aistats2019/0-AISTATS2019-slides-zhi-hua_zhou.pdf>`__
* Source Code: `[GitHub] <https://github.com/LAMDA-NJU/Deep-Forest>`__ | `[Gitee] <https://gitee.com/lamda-nju/deep-forest>`__

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
   Report from Users <report_from_users>

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics

   Model Architecture <./advanced_topics/architecture>
   Use Customized Estimators <./advanced_topics/use_customized_estimator>

.. toctree::
   :maxdepth: 1
   :caption: For Developers

   Contributors <contributors>
   Changelog <changelog>

.. toctree::
   :maxdepth: 1
   :caption: About

   About Us <http://www.lamda.nju.edu.cn/MainPage.ashx>
   Related Software <related_software>
