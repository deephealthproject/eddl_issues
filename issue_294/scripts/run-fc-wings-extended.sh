#!/bin/bash


for output_type in "sigmoid" "softmax"
do
    for device_type in "cpu" "gpu"
    do

        rm -f report-cpp.log

        scripts/run-fc-wings.sh --epochs 20 --${output_type}-at-output --${device_type} --create

        for i in {1..5}
        do
            scripts/run-fc-wings.sh --epochs 20 --${output_type}-at-output --${device_type} 
        done

        mv report-cpp.log report-cpp-${output_type}-${device_type}.log
    done
done
