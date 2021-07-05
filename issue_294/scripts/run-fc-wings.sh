#!/bin/bash

cd build
make
cd ..

export LD_LIBRARY_PATH="${HOME}/projects/eddl/build/cmake/third_party/protobuf/lib:${CONDA_PREFIX}/lib"
#export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
#export LD_PRELOAD="${CONDA_PREFIX}/lib/libasan.so"
LSAN_OPTIONS="verbosity=1:log_threads=1"

executable="build/bin/fully_conn_wings.exe"

#ldd ${executable}
#exit 0

${executable} $*
#gdb ${executable}
#valgrind --leak-check=full --track-origins=yes -v ${executable} --reduced $*
