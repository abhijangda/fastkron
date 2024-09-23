#!/usr/bin/bash

export PATH="/opt/python/$1-$1/bin:$PATH"
cd $2
rm -rf build/
python setup.py bdist_wheel