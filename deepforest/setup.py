import os
import numpy
from distutils.version import LooseVersion
from numpy.distutils.misc_util import Configuration


CYTHON_MIN_VERSION = "0.24"


def configuration(parent_package="", top_path=None):

    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config = Configuration("deepforest", parent_package, top_path)
    config.add_subpackage("tree")

    config.add_extension(
        "_forest",
        sources=["_forest.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    )

    config.add_extension(
        "_cutils",
        sources=["_cutils.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    )

    msg = (
        "Please install cython with a version >= {} in order to build a"
        " deepforest development version."
    )
    msg = msg.format(CYTHON_MIN_VERSION)

    try:
        import Cython

        if LooseVersion(Cython.__version__) < CYTHON_MIN_VERSION:
            msg += " Your version of Cython is {}.".format(Cython.__version__)
            raise ValueError(msg)
        from Cython.Build import cythonize
    except ImportError as exc:
        exc.args += (msg,)
        raise

    config.ext_modules = cythonize(config.ext_modules)

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())
