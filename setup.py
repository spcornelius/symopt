#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
from itertools import chain

import setuptools

pkg_name = 'sympot'
license = 'MIT'
version = '0.1.0'

extras_req = {
    'testing': ['pytest', 'pytest-pep8']
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

with open("README.md", "r") as f:
    long_description = f.read()

if __name__ == '__main__':
    setup(
        name=pkg_name.lower(),
        version=version,
        description="(Non)linear optimization with symbolically-defined "
                    "objective/constraints",
        long_description=long_description,
        long_description_content_type="text/x-rst",
        author="Sean P. Cornelius",
        author_email="spcornelius@gmail.com",
        url="https://github.com/spcornelius/symopt",
        license=license,
        packages=setuptools.find_packages(),
        install_requires=['numpy', 'scipy', 'sympy', 'orderedset'],
        extras_require=extras_req,
        python_requires='>=3.6',
        classifiers=classifiers)
