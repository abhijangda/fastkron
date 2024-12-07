#!/usr/bin/bash

export PATH="/opt/python/cp311-cp311/bin:$PATH"
/opt/python/cp311-cp311/bin/pip install setuptools-scm
cd $1
rm -rf build/
BUILD_ANY_WHEEL=1 python3.11 setup.py bdist_wheel