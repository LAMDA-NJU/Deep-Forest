Installation Guide
==================

Stable Version
--------------

The stable version is available via `PyPI <https://pypi.org/>`__ using:

.. code-block:: bash

    $ pip install deep-forest

The package is portable and with very few package dependencies. It is recommended to use the package environment from `Anaconda <https://www.anaconda.com/>`__ since it already installs all required packages.


Building from Source
--------------------

Building from source is required to work on a contribution (bug fix, new feature, code or documentation improvement).

- **Use Git to check out the latest source from the repository on Github:**

.. code-block:: bash

    $ git clone https://github.com/LAMDA-NJU/Deep-Forest.git
    $ cd Deep-Forest

- **Install a C compiler for your platform.**

.. note::

    The compiler is used to compile the Cython files in the package. Please refer to `Installing Cython <https://cython.readthedocs.io/en/latest/src/quickstart/install.html>`__ for details on choosing the compiler.

- **Optional (but recommended): create and activate a dedicated virtual environment or conda environment.**

- **Build the project with pip in the editable mode:**

.. code-block:: bash

    $ pip install --verbose -e .

.. note::

    The key advantage of the editable mode is that there is no need to re-install the entire package if you have modified a Python file. However, you still have to run the ``pip install --verbose -e .`` command again if the source code of a Cython file is updated (ending with .pyx or .pxd).

- **Optional: run the tests on the module:**

.. code-block:: bash

    $ cd tests
    % pytest

Acknowledgement
---------------

The installation instructions were adapted from Scikit-Learn’s `advanced installation instructions <https://scikit-learn.org/stable/developers/advanced_installation.html>`__.