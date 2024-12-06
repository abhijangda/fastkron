#!/usr/bin/bash

export PATH="/opt/python/$1-$1/bin:$PATH"
cd $2
/opt/python/$1-$1/bin/pip install setuptools-scm
rm -rf build/
python setup.py bdist_wheel