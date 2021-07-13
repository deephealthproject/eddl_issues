#!/bin/bash


epochs=50

for optimizer in "sgd" "rmsprop" "adam"
do
    for output_type in "sigmoid" "softmax"
    do
        for device_type in "cpu" "gpu"
        do
            title="${output_type} ${device_type} ${optimizer} ${epochs} epochs"
            python python/results-visualisation.py --title "${title}" logs/report-cpp-*-${output_type}-${device_type}-${optimizer}-epochs-${epochs}.log
        done
    done
done

