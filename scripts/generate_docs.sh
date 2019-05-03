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
NEW="'sphinx.ext.viewcode', 'sphinx.ext.autosummary', 'numpydoc', 'm2r'"
sed -i "" "s/$MATCH/$NEW/g" docs/conf.py
sed -i "" "s/alabaster/sphinx_rtd_theme/g" docs/conf.py

echo "numpydoc_class_members_toctree = False" >>docs/conf.py
ABS_REPO_PATH=$(unset CDPATH && cd "$(dirname "$0")/.." && echo $PWD)
if [[ ! -d docs/_build/html ]]; then
    mkdir docs/_build/html
fi

( cd docs; PYTHONPATH=$ABS_REPO_PATH make html >_build/html/build.log )