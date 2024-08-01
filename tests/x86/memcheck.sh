#!/usr/bin/sh

OMP_NUM_THREADS=1 valgrind --leak-check=full --show-leak-kinds=all --xml=yes --xml-file=l.xml $1
