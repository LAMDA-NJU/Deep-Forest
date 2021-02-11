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

.. |MajorFeature| replace:: :raw-html:`<span class="badge badge-success">Major Feature</span>` :raw-latex:`{\small\sc [Major Feature]}`
.. |Feature| replace:: :raw-html:`<span class="badge badge-success">Feature</span>` :raw-latex:`{\small\sc [Feature]}`
.. |Efficiency| replace:: :raw-html:`<span class="badge badge-info">Efficiency</span>` :raw-latex:`{\small\sc [Efficiency]}`
.. |Enhancement| replace:: :raw-html:`<span class="badge badge-info">Enhancement</span>` :raw-latex:`{\small\sc [Enhancement]}`
.. |Fix| replace:: :raw-html:`<span class="badge badge-danger">Fix</span>` :raw-latex:`{\small\sc [Fix]}`
.. |API| replace:: :raw-html:`<span class="badge badge-warning">API Change</span>` :raw-latex:`{\small\sc [API Change]}`

- |Feature| support configurable criterion (`#28 <https://github.com/LAMDA-NJU/Deep-Forest/issues/28>`__) @tczhao
- |Feature| support regression prediction (`#25 <https://github.com/LAMDA-NJU/Deep-Forest/issues/25>`__) @tczhao
- |Fix| fix accepted data types on the :obj:`binner` (`#23 <https://github.com/LAMDA-NJU/Deep-Forest/pull/23>`__) @xuyxu
- |Feature| implement the :meth:`get_forest` method for efficient indexing (`#22 <https://github.com/LAMDA-NJU/Deep-Forest/pull/22>`__) @xuyxu
- |Feature| support class label encoding (`#18 <https://github.com/LAMDA-NJU/Deep-Forest/pull/18>`__) @NiMaZi
- |Feature| support sample weight in :meth:`fit` (`#7 <https://github.com/LAMDA-NJU/Deep-Forest/pull/7>`__) @tczhao
- |Feature| configurable predictor parameter (`#9 <https://github.com/LAMDA-NJU/Deep-Forest/issues/10>`__) @tczhao
- |Enhancement| add base class ``BaseEstimator`` and ``ClassifierMixin`` (`#8 <https://github.com/LAMDA-NJU/Deep-Forest/pull/8>`__) @pjgao
