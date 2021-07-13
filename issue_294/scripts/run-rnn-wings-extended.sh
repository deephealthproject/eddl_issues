#!/bin/bash

cd build
make
cd ..

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"

executable="build/bin/rnn_wings.exe"

batch_size=50
epochs=50
repetitions=10


for optimizer in "sgd" "rmsprop" "adam"
do
    for output_type in "sigmoid" "softmax"
    do
        for device_type in "cpu" "gpu"
        do
            rm -f report-cpp.log

            ${executable} --epochs ${epochs} --${output_type}-at-output --${device_type} --optimizer ${optimizer} --create #--repetitions ${repetitions}
            mv report-cpp.log logs/report-cpp-created-${output_type}-${device_type}-${optimizer}-epochs-${epochs}.log

            for i in {1..10}
            do
                ${executable} --epochs ${epochs} --${output_type}-at-output --${device_type} --optimizer ${optimizer}
            done
            mv report-cpp.log logs/report-cpp-via-onnx-${output_type}-${device_type}-${optimizer}-epochs-${epochs}.log
        done
    done
done
