#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
from itertools import chain

import os
import setuptools
import re

pkg_name = 'symopt'
license = 'MIT'
url = "https://github.com/spcornelius/symopt"
author = "Sean P. Cornelius"
author_email = "spcornelius@gmail.com"


def _path_under_setup(*args):
    return os.path.join(os.path.dirname(__file__), *args)


version_file = _path_under_setup(pkg_name, "_version.py")
readme_file = "README.md"

extras_req = {
    'testing': ['pytest', 'pytest-pep8'],
    'docs': ['sphinx <= 1.8.4', 'sphinx_rtd_theme <= 2.4', 'numpydoc', 'm2r']
}

extras_req['all'] = list(chain(*extras_req.values()))

classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Mathematics",
]

match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                  open(version_file, "rt").read(),
                  re.M)
if match:
    version = match.group(1)
else:
    raise RuntimeError(
        "Unable to find version string in %s." % (version_file,))

with open(readme_file, "r") as f:
    long_description = f.read()

if __name__ == '__main__':
    setup(
        name=pkg_name.lower(),
        version=version,
        description="Easy (non)linear optimization in Python with "
                    "symbolically-defined objective/constraints",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author=author,
        author_email=author_email,
        url=url,
        license=license,
        packages=setuptools.find_packages(),
        install_requires=['numpy', 'scipy', 'sympy', 'orderedset'],
        extras_require=extras_req,
        python_requires='>=3.6',
        classifiers=classifiers)
