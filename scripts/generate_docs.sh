#!/bin/bash -xe
#
# Usage:
#
#    $ ./scripts/generate_docs.sh
#
NARGS=$#
PKG=$(find . -maxdepth 2 -name __init__.py -print0 | xargs -0 -n1 dirname | xargs basename)
AUTHOR=$(head -n 1 AUTHORS)
sphinx-apidoc --full --force -A "$AUTHOR" --module-first --doc-version=$(python setup.py --version) -F -o docs $PKG/ $(find . -type d -name tests)
cat <<EOF >>docs/index.rst

Overview
========
.. mdinclude:: README.md

EOF
MATCH="'sphinx.ext.viewcode'"
NEW="'sphinx.ext.viewcode', 'sphinx.ext.autosummary', 'sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'numpydoc', 'm2r'"
sed -i "s/$MATCH/$NEW/g" docs/conf.py
sed -i "s/alabaster/sphinx_rtd_theme/g" docs/conf.py

echo "default_role = 'obj'" >> docs/conf.py
echo "autoclass_content = 'both'" >> docs/conf.py # include both class docstring and __init__
echo "autodoc_default_flags = ['members']" >> docs/conf.py
echo "autosummary_generate = True" >> docs/conf.py
echo "numpydoc_class_members_toctree = False" >>docs/conf.py
echo "numpydoc_attributes_as_param_list = False" >>docs/conf.py
echo "pygments_style = 'sphinx'" >>docs/conf.py


echo "intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                             'numpy': ('https://docs.scipy.org/doc/numpy', None),
                             'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
                             'orderedset': ('https://orderedset.readthedocs.io/en/latest/', None),
                             'sympy': ('https://docs.sympy.org/latest/', None)
                             }" >> docs/conf.py

if [[ ! -d docs/_build/html ]]; then
    mkdir docs/_build/html
fi
( cd docs; PYTHONPATH=$ABS_REPO_PATH make html >_build/html/build.log )