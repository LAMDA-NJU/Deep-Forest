Welcome to Deep Forest's Documentation
======================================

**Deep Forest** is a general ensemble framework that uses tree-based ensemble algorithms such as Random Forest. It is designed to have the following advantages:

- **Powerful**: Better accuracy than existing tree-based ensemble methods.
- **Easy to Use**: Less efforts on tunning parameters.
- **Efficient**: Fast training speed and high efficiency.
- **Scalable**: Capable of handling large-scale data.

For a quick start, please refer to `How to Get Started <./how_to_get_started.html>`__. For a detailed guidance on parameter tunning, please refer to `Parameters Tunning <./parameters_tunning.html>`__.

The package is actively being developed. The goal is to provide users from the industrial and academic community with a third option on tree-based ensemble methods apart from Random Forest and Gradient Boosting Decision Tree. To achieve this, any help would be welcomed. Please check the `Homepage <https://github.com/LAMDA-NJU/DF21>`__ for details.

Installation
------------

The package is available via PyPI using:

.. code-block:: bash

    $ pip install deep-forest

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

Resources
---------

* Deep Forest: `[Paper] <https://arxiv.org/pdf/1702.08835.pdf>`__
* Keynote at AISTATS 2019: `[Slides] <https://aistats.org/aistats2019/0-AISTATS2019-slides-zhi-hua_zhou.pdf>`__

Reference
---------

.. code-block:: latex

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
