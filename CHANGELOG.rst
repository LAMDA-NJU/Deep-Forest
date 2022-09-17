Changelog
=========

+---------------+-----------------------------------------------------------+
| Badge         | Meaning                                                   |
+===============+===========================================================+
| |Feature|     | Add something that cannot be achieved before.             |
+---------------+-----------------------------------------------------------+
| |Efficiency|  | Improve the efficiency on the computation or memory.      |
+---------------+-----------------------------------------------------------+
| |Enhancement| | Miscellaneous minor improvements.                         |
+---------------+-----------------------------------------------------------+
| |Fix|         | Fix up something that does not work as expected.          |
+---------------+-----------------------------------------------------------+
| |API|         | You will need to change the code to have the same effect. |
+---------------+-----------------------------------------------------------+

Version 0.1.*
-------------

.. role:: raw-html(raw)
   :format: html

.. role:: raw-latex(raw)
   :format: latex

.. |Feature| replace:: :raw-html:`<span class="badge badge-success">Feature</span>` :raw-latex:`{\small\sc [Feature]}`
.. |Efficiency| replace:: :raw-html:`<span class="badge badge-info">Efficiency</span>` :raw-latex:`{\small\sc [Efficiency]}`
.. |Enhancement| replace:: :raw-html:`<span class="badge badge-primary">Enhancement</span>` :raw-latex:`{\small\sc [Enhancement]}`
.. |Fix| replace:: :raw-html:`<span class="badge badge-danger">Fix</span>` :raw-latex:`{\small\sc [Fix]}`
.. |API| replace:: :raw-html:`<span class="badge badge-warning">API Change</span>` :raw-latex:`{\small\sc [API Change]}`

- |Feature| support the latest version of scikit-learn and drop support on python 3.6 (`#115 <https://github.com/LAMDA-NJU/Deep-Forest/pull/115>`__) @xuyxu
- |Feature| |API| add support on :obj:`pandas.DataFrame` for ``X`` and ``y`` (`#86 <https://github.com/LAMDA-NJU/Deep-Forest/pull/86>`__) @IncubatorShokuhou
- |Fix| fix missing functionality of :meth:`_set_n_trees` @xuyxu
- |Fix| |API| add docstrings for parameter ``bin_type`` (`#74 <https://github.com/LAMDA-NJU/Deep-Forest/pull/74>`__) @xuyxu
- |Feature| |API| recover the parameter ``min_samples_split`` (`#73 <https://github.com/LAMDA-NJU/Deep-Forest/pull/73>`__) @xuyxu
- |Fix| fix the breakdown under the corner case where no internal node exists (`#70 <https://github.com/LAMDA-NJU/Deep-Forest/pull/70>`__) @xuyxu
- |Feature| support python 3.9 (`#69 <https://github.com/LAMDA-NJU/Deep-Forest/pull/69>`__) @xuyxu
- |Fix| fix inconsistency on array shape for :obj:`CascadeForestRegressor` in customized mode (`#67 <https://github.com/LAMDA-NJU/Deep-Forest/pull/67>`__) @xuyxu
- |Fix| fix missing sample indices for parameter ``sample_weight`` in :obj:`KFoldWrapper` (`#48 <https://github.com/LAMDA-NJU/Deep-Forest/pull/64>`__) @xuyxu
- |Feature| |API| add support on customized estimators (`#48 <https://github.com/LAMDA-NJU/Deep-Forest/pull/48>`__) @xuyxu
- |Enhancement| improve target checks for :obj:`CascadeForestRegressor` (`#53 <https://github.com/LAMDA-NJU/Deep-Forest/pull/53>`__) @chendingyan
- |Fix| fix the prediction workflow with only one cascade layer (`#56 <https://github.com/LAMDA-NJU/Deep-Forest/pull/56>`__) @xuyxu
- |Fix| fix inconsistency on predictor name (`#52 <https://github.com/LAMDA-NJU/Deep-Forest/pull/52>`__) @xuyxu
- |Feature| add official support for ManyLinux-aarch64 (`#47 <https://github.com/LAMDA-NJU/Deep-Forest/pull/47>`__) @xuyxu
- |Fix| fix accepted types of target for :obj:`CascadeForestRegressor` (`#44 <https://github.com/LAMDA-NJU/Deep-Forest/pull/44>`__) @xuyxu
- |Feature| |API| add multi-output support for :obj:`CascadeForestRegressor` (`#40 <https://github.com/LAMDA-NJU/Deep-Forest/pull/40>`__) @Alex-Medium
- |Feature| |API| add layer-wise feature importances (`#39 <https://github.com/LAMDA-NJU/Deep-Forest/pull/39>`__) @xuyxu
- |Feature| |API| add scikit-learn backend (`#36 <https://github.com/LAMDA-NJU/Deep-Forest/pull/36>`__) @xuyxu
- |Feature| add official support for Mac-OS (`#34 <https://github.com/LAMDA-NJU/Deep-Forest/pull/34>`__) @T-Allen-sudo
- |Feature| |API| support configurable criterion (`#28 <https://github.com/LAMDA-NJU/Deep-Forest/issues/28>`__) @tczhao
- |Feature| |API| support regression prediction (`#25 <https://github.com/LAMDA-NJU/Deep-Forest/issues/25>`__) @tczhao
- |Fix| fix accepted data types on the :obj:`binner` (`#23 <https://github.com/LAMDA-NJU/Deep-Forest/pull/23>`__) @xuyxu
- |Feature| |API| implement the :meth:`get_estimator` method for efficient indexing (`#22 <https://github.com/LAMDA-NJU/Deep-Forest/pull/22>`__) @xuyxu
- |Feature| support class label encoding (`#18 <https://github.com/LAMDA-NJU/Deep-Forest/pull/18>`__) @NiMaZi
- |Feature| |API| support sample weight in :meth:`fit` (`#7 <https://github.com/LAMDA-NJU/Deep-Forest/pull/7>`__) @tczhao
- |Feature| |API| configurable predictor parameter (`#9 <https://github.com/LAMDA-NJU/Deep-Forest/issues/10>`__) @tczhao
- |Enhancement| add base class ``BaseEstimator`` and ``ClassifierMixin`` (`#8 <https://github.com/LAMDA-NJU/Deep-Forest/pull/8>`__) @pjgao
