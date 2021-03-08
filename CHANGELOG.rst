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
.. |Enhancement| replace:: :raw-html:`<span class="badge badge-info">Enhancement</span>` :raw-latex:`{\small\sc [Enhancement]}`
.. |Fix| replace:: :raw-html:`<span class="badge badge-danger">Fix</span>` :raw-latex:`{\small\sc [Fix]}`
.. |API| replace:: :raw-html:`<span class="badge badge-warning">API Change</span>` :raw-latex:`{\small\sc [API Change]}`

- |Enhancement| improve target checks for :obj:`CascadeForestRegressor` (`#53 <https://github.com/LAMDA-NJU/Deep-Forest/pull/53>`__) @chendingyan
- |Fix| fix the prediction workflow with only one cascade layer (`#56 <https://github.com/LAMDA-NJU/Deep-Forest/pull/56>`__) @xuyxu
- |Fix| fix inconsistency on predictor name (`#52 <https://github.com/LAMDA-NJU/Deep-Forest/pull/52>`__) @xuyxu
- |Feature| add official support for ManyLinux-aarch64 (`#47 <https://github.com/LAMDA-NJU/Deep-Forest/pull/47>`__) @xuyxu
- |Fix| fix accepted types of target for :obj:`CascadeForestRegressor` (`#44 <https://github.com/LAMDA-NJU/Deep-Forest/pull/44>`__) @xuyxu
- |Feature| add multi-output support for :obj:`CascadeForestRegressor` (`#40 <https://github.com/LAMDA-NJU/Deep-Forest/pull/40>`__) @Alex-Medium
- |Feature| add layer-wise feature importances (`#39 <https://github.com/LAMDA-NJU/Deep-Forest/pull/39>`__) @xuyxu
- |Feature| add scikit-learn backend (`#36 <https://github.com/LAMDA-NJU/Deep-Forest/pull/36>`__) @xuyxu
- |Feature| add official support for Mac-OS (`#34 <https://github.com/LAMDA-NJU/Deep-Forest/pull/34>`__) @T-Allen-sudo
- |Feature| support configurable criterion (`#28 <https://github.com/LAMDA-NJU/Deep-Forest/issues/28>`__) @tczhao
- |Feature| support regression prediction (`#25 <https://github.com/LAMDA-NJU/Deep-Forest/issues/25>`__) @tczhao
- |Fix| fix accepted data types on the :obj:`binner` (`#23 <https://github.com/LAMDA-NJU/Deep-Forest/pull/23>`__) @xuyxu
- |Feature| implement the :meth:`get_forest` method for efficient indexing (`#22 <https://github.com/LAMDA-NJU/Deep-Forest/pull/22>`__) @xuyxu
- |Feature| support class label encoding (`#18 <https://github.com/LAMDA-NJU/Deep-Forest/pull/18>`__) @NiMaZi
- |Feature| support sample weight in :meth:`fit` (`#7 <https://github.com/LAMDA-NJU/Deep-Forest/pull/7>`__) @tczhao
- |Feature| configurable predictor parameter (`#9 <https://github.com/LAMDA-NJU/Deep-Forest/issues/10>`__) @tczhao
- |Enhancement| add base class ``BaseEstimator`` and ``ClassifierMixin`` (`#8 <https://github.com/LAMDA-NJU/Deep-Forest/pull/8>`__) @pjgao
