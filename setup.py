import os
import sys
from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize
import numpy

# Project Information
DISTNAME = "deep-forest"
DESCRIPTION = "Deep Forest"
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Yi-Xuan Xu"
MAINTAINER_EMAIL = "xuyx@lamda.nju.edu.cn"
URL = "https://github.com/LAMDA-NJU/Deep-Forest"
VERSION = "0.1.7"

libraries = []
if os.name == "posix":
    libraries.append("m")

extensions = [
    Extension(
        "deepforest._forest",
        ["deepforest/_forest.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
    Extension(
        "deepforest._cutils",
        ["deepforest/_cutils.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
    Extension(
        "deepforest.tree._tree",
        ["deepforest/tree/_tree.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
    Extension(
        "deepforest.tree._splitter",
        ["deepforest/tree/_splitter.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
    Extension(
        "deepforest.tree._criterion",
        ["deepforest/tree/_criterion.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
    Extension(
        "deepforest.tree._utils",
        ["deepforest/tree/_utils.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
]

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        packages=find_packages(),
        include_package_data=True,
        description=DESCRIPTION,
        url=URL,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        zip_safe=False,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Unix",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        python_requires=">=3.7",
        install_requires=[
            "numpy>=1.14.6",
            "scipy>=1.1.0",
            "joblib>=0.11",
            "scikit-learn>=1.0",
        ],
        setup_requires=["cython"],
        ext_modules=cythonize(extensions),
    )