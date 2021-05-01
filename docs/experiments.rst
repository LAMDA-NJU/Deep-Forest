Experiments
===========

Baseline
--------
For all experiments, we used 5 popular tree-based ensemble methods as baselines. Details on the baselines are listed in the following table:

+------------------+---------------------------------------------------------------+
|       Name       |                          Introduction                         |
+==================+===============================================================+
| `Random Forest`_ | An efficient implementation of Random Forest in Scikit-Learn  |
+------------------+---------------------------------------------------------------+
|     `HGBDT`_     |             Histogram-based GBDT in Scikit-Learn              |
+------------------+---------------------------------------------------------------+
| `XGBoost EXACT`_ |                The vanilla version of XGBoost                 |
+------------------+---------------------------------------------------------------+
|  `XGBoost HIST`_ |          The histogram optimized version of XGBoost           |
+------------------+---------------------------------------------------------------+
|    `LightGBM`_   |                Light Gradient Boosting Machine                |
+------------------+---------------------------------------------------------------+

Environment
-----------
For all experiments, we used a single linux server. Details on the specifications are listed in the table below. All processors were used for training and evaluating.

+------------------+-----------------+--------+
|        OS        |       CPU       | Memory |
+==================+=================+========+
| Ubuntu 18.04 LTS |   Xeon E-2288G  | 128GB  |
+------------------+-----------------+--------+

Setting
-------
We kept the number of decision trees the same across all baselines, while remaining hyper-parameters were set to their default values. Running scripts on reproducing all experiment results are available, please refer to this `Repo`_.

Classification
--------------

Dataset
*******

We have collected a number of datasets for both binary and multi-class classification, as listed in the table below. They were selected based on the following criteria:

- Publicly available and easy to use;
- Cover different application areas;
- Reflect high diversity in terms of the number of samples, features, and classes.

As a result, some baselines may fail on datasets with too many samples or features. Such cases are indicated by ``N/A`` in all tables below.

+------------------+------------+-----------+------------+-----------+
|       Name       | # Training | # Testing | # Features | # Classes |
+==================+============+===========+============+===========+
|     `ijcnn1`_    |   49,990   |   91,701  |     22     |     2     |
+------------------+------------+-----------+------------+-----------+
|   `pendigits`_   |    7,494   |   3,498   |     16     |     10    |
+------------------+------------+-----------+------------+-----------+
|     `letter`_    |   15,000   |   5,000   |     16     |     26    |
+------------------+------------+-----------+------------+-----------+
|   `connect-4`_   |   67,557   |   20,267  |     126    |     3     |
+------------------+------------+-----------+------------+-----------+
|     `sector`_    |    6,412   |   3,207   |   55,197   |    105    |
+------------------+------------+-----------+------------+-----------+
|    `covtype`_    |   406,708  |  174,304  |     54     |     7     |
+------------------+------------+-----------+------------+-----------+
|      `susy`_     |  4,500,000 |  500,000  |     18     |     2     |
+------------------+------------+-----------+------------+-----------+
|     `higgs`_     | 10,500,000 |  500,000  |     28     |     2     |
+------------------+------------+-----------+------------+-----------+
|      `usps`_     |    7,291   |   2,007   |     256    |     10    |
+------------------+------------+-----------+------------+-----------+
|     `mnist`_     |   60,000   |   10,000  |     784    |     10    |
+------------------+------------+-----------+------------+-----------+
| `fashion mnist`_ |   60,000   |   10,000  |     784    |     10    |
+------------------+------------+-----------+------------+-----------+

Classification Accuracy
***********************

The table below shows the testing accuracy of each method, with the best result on each dataset **bolded**. Each experiment was conducted over 5 independently trials, and the average result was reported.

+---------------+-------+-------+-----------+-----------+-----------+-------------+
|      Name     |   RF  | HGBDT | XGB EXACT |  XGB HIST |  LightGBM | Deep Forest |
+===============+=======+=======+===========+===========+===========+=============+
|     ijcnn1    | 98.07 | 98.43 |   98.20   |   98.23   | **98.61** |    98.16    |
+---------------+-------+-------+-----------+-----------+-----------+-------------+
|   pendigits   | 96.54 | 96.34 |   96.60   |   96.60   |   96.17   |  **97.50**  |
+---------------+-------+-------+-----------+-----------+-----------+-------------+
|     letter    | 95.39 | 91.56 |   90.80   |   90.82   |   88.94   |  **95.92**  |
+---------------+-------+-------+-----------+-----------+-----------+-------------+
|   connect-4   | 70.18 | 70.88 |   71.57   |   71.57   |   70.31   |  **72.05**  |
+---------------+-------+-------+-----------+-----------+-----------+-------------+
|     sector    | 85.62 |  N/A  |   66.01   |   65.61   |   63.24   |  **86.74**  |
+---------------+-------+-------+-----------+-----------+-----------+-------------+
|    covtype    | 73.73 | 64.22 |   66.15   |   66.70   |   65.00   |  **74.27**  |
+---------------+-------+-------+-----------+-----------+-----------+-------------+
|      susy     | 80.19 | 80.31 |   80.32   | **80.35** |   80.33   |    80.18    |
+---------------+-------+-------+-----------+-----------+-----------+-------------+
|     higgs     |  N/A  | 74.95 |   75.85   |   76.00   |   74.97   |  **76.46**  |
+---------------+-------+-------+-----------+-----------+-----------+-------------+
|      usps     | 93.79 | 94.32 |   93.77   |   93.37   |   93.97   |  **94.67**  |
+---------------+-------+-------+-----------+-----------+-----------+-------------+
|     mnist     | 97.20 | 98.35 |   98.07   |   98.14   | **98.42** |    98.11    |
+---------------+-------+-------+-----------+-----------+-----------+-------------+
| fashion mnist | 87.87 | 87.02 |   90.74   |   90.80   | **90.81** |    89.66    |
+---------------+-------+-------+-----------+-----------+-----------+-------------+

Runtime
*******

Runtime in seconds reported in the table below covers both the training stage and evaluating stage.

+---------------+---------+--------+-----------+----------+----------+-------------+
|      Name     |    RF   |  HGBDT | XGB EXACT | XGB HIST | LightGBM | Deep Forest |
+===============+=========+========+===========+==========+==========+=============+
|     ijcnn1    |   9.60  |  6.84  |   11.24   |   1.90   |   1.99   |     8.37    |
+---------------+---------+--------+-----------+----------+----------+-------------+
|   pendigits   |   1.26  |  5.12  |    0.39   |   0.26   |   0.46   |     2.21    |
+---------------+---------+--------+-----------+----------+----------+-------------+
|     letter    |   0.76  |  1.30  |    0.34   |   0.17   |   0.19   |     2.84    |
+---------------+---------+--------+-----------+----------+----------+-------------+
|   connect-4   |   5.17  |  7.54  |   13.26   |   3.19   |   1.12   |    10.73    |
+---------------+---------+--------+-----------+----------+----------+-------------+
|     sector    |  292.15 |   N/A  |   632.27  |  593.35  |  18.83   |    521.68   |
+---------------+---------+--------+-----------+----------+----------+-------------+
|    covtype    |  84.00  |  2.56  |   58.43   |  11.62   |   3.96   |    164.18   |
+---------------+---------+--------+-----------+----------+----------+-------------+
|      susy     | 1429.85 |  59.09 |  1051.54  |  44.85   |  34.40   |   1866.48   |
+---------------+---------+--------+-----------+----------+----------+-------------+
|     higgs     |   N/A   | 523.74 |  7532.70  |  267.64  |  209.65  |   7307.44   |
+---------------+---------+--------+-----------+----------+----------+-------------+
|      usps     |   9.28  |  8.73  |    9.43   |   5.78   |   9.81   |     6.08    |
+---------------+---------+--------+-----------+----------+----------+-------------+
|     mnist     |  590.81 | 229.91 |  1156.64  |  762.40  |  233.94  |    599.55   |
+---------------+---------+--------+-----------+----------+----------+-------------+
| fashion mnist |  735.47 |  32.86 |  1403.44  | 2061.80  |  428.37  |    661.05   |
+---------------+---------+--------+-----------+----------+----------+-------------+

Some observations are listed as follow:

* Histogram-based GBDT (e.g., :class:`HGBDT`, :class:`XGB HIST`, :class:`LightGBM`) are typically faster mainly because decision tree in GBDT tends to have a much smaller tree depth;
* With the number of input dimensions increasing (e.g., on mnist and fashion-mnist), random forest and deep forest can be faster.

Regression
----------

Dataset
*******

We have also collected four datasets on univariate regression for a comparison on the regression problem.

+------------------+------------+-----------+------------+
|       Name       | # Training | # Testing | # Features |
+==================+============+===========+============+
|      `wine`_     |    1,071   |    528    |     11     |
+------------------+------------+-----------+------------+
|     `abalone`_   |    2,799   |   1,378   |      8     |
+------------------+------------+-----------+------------+
|    `cpusmall`_   |    5,489   |   2,703   |     12     |
+------------------+------------+-----------+------------+
|     `boston`_    |     379    |    127    |     13     |
+------------------+------------+-----------+------------+
|    `diabetes`_   |     303    |    139    |     10     |
+------------------+------------+-----------+------------+

Testing Mean Squared Error
**************************

The table below shows the testing mean squared error of each method, with the best result on each dataset **bolded**. Each experiment was conducted over 5 independently trials, and the average result was reported.

+----------+-----------+---------+-----------+----------+----------+-------------+
|   Name   |     RF    |  HGBDT  | XGB EXACT | XGB HIST | LightGBM | Deep Forest |
+==========+===========+=========+===========+==========+==========+=============+
|   wine   |    0.35   |   0.40  |    0.41   |   0.41   |   0.39   |   **0.34**  |
+----------+-----------+---------+-----------+----------+----------+-------------+
|  abalone |    4.79   |   5.40  |    5.73   |   5.75   |   5.60   |   **4.66**  |
+----------+-----------+---------+-----------+----------+----------+-------------+
| cpusmall |    8.31   |   9.01  |    9.86   |   11.82  |   8.99   |   **7.15**  |
+----------+-----------+---------+-----------+----------+----------+-------------+
|  boston  | **16.61** |  20.68  |   20.61   |   19.65  |   20.27  |    19.87    |
+----------+-----------+---------+-----------+----------+----------+-------------+
| diabetes |  3796.62  | 4333.66 |  4337.15  |  4303.96 |  4435.95 | **3431.01** |
+----------+-----------+---------+-----------+----------+----------+-------------+

Runtime
*******

Runtime in seconds reported in the table below covers both the training stage and evaluating stage.

+----------+------+-------+-----------+----------+----------+-------------+
|   Name   |  RF  | HGBDT | XGB EXACT | XGB HIST | LightGBM | Deep Forest |
+==========+======+=======+===========+==========+==========+=============+
|   wine   | 0.76 |  2.88 |    0.30   |   0.30   |   0.30   |     1.26    |
+----------+------+-------+-----------+----------+----------+-------------+
|  abalone | 0.53 |  1.57 |    0.47   |   0.50   |   0.17   |     1.29    |
+----------+------+-------+-----------+----------+----------+-------------+
| cpusmall | 1.87 |  3.59 |    1.71   |   1.25   |   0.36   |     2.06    |
+----------+------+-------+-----------+----------+----------+-------------+
|  boston  | 0.70 |  1.75 |    0.19   |   0.22   |   0.20   |     1.45    |
+----------+------+-------+-----------+----------+----------+-------------+
| diabetes | 0.37 |  0.66 |    0.14   |   0.18   |   0.06   |     1.09    |
+----------+------+-------+-----------+----------+----------+-------------+

.. _`Random Forest`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

.. _`HGBDT`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html

.. _`XGBoost EXACT`: https://xgboost.readthedocs.io/en/latest/index.html

.. _`XGBoost HIST`: https://xgboost.readthedocs.io/en/latest/index.html

.. _`LightGBM`: https://lightgbm.readthedocs.io/en/latest/

.. _`Repo`: https://github.com/xuyxu/deep_forest_benchmarks

.. _`ijcnn1`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1

.. _`pendigits`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#pendigits

.. _`letter`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#letter

.. _`connect-4`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#connect-4

.. _`sector`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#sector

.. _`covtype`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#covtype

.. _`susy`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#SUSY

.. _`higgs`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#HIGGS

.. _`usps`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps

.. _`mnist`: https://keras.io/api/datasets/mnist/

.. _`fashion mnist`: https://keras.io/api/datasets/fashion_mnist/

.. _`wine`: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

.. _`abalone`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#abalone

.. _`cpusmall`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cpusmall

.. _`boston`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html

.. _`diabetes`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
