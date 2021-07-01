#!/bin/bash

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
#export LD_PRELOAD="${CONDA_PREFIX}/lib/libasan.so"

build/rnn_wings.exe $*
#valgrind --leak-check=full --track-origins=yes build/rnn_wings.exe --reduced $*
