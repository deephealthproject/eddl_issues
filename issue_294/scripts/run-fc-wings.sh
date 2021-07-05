#!/bin/bash

cd build
make
cd ..

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
#export LD_PRELOAD="${CONDA_PREFIX}/lib/libasan.so"

build/fully_conn_wings.exe $*
#valgrind --leak-check=full --track-origins=yes build/fully_conn_wings.exe --reduced $*
