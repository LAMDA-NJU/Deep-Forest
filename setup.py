import os
import sys
from setuptools import find_packages
from numpy.distutils.core import setup


# Project Information
DISTNAME = "deep-forest"
DESCRIPTION = "Deep Forest"
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Yi-Xuan Xu"
MAINTAINER_EMAIL = "xuyx@lamda.nju.edu.cn"
URL = "https://github.com/LAMDA-NJU/Deep-Forest"
VERSION = "0.1.5"


def configuration(parent_package="", top_path=None):

    if os.path.exists("MANIFEST"):
        os.remove("MANIFEST")

    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.add_subpackage("deepforest")

    return config


if __name__ == "__main__":

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(
        configuration=configuration,
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
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        python_requires=">=3.6",
        install_requires=[
            "numpy>=1.16.0,<1.20.0",
            "scipy>=0.19.1",
            "joblib>=0.11",
            "scikit-learn>=0.22",
        ],
        setup_requires=["cython"],
    )
