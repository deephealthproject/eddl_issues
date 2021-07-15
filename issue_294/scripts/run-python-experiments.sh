#!/bin/bash

rm -f report-pyeddl.log report-keras.log

for i in {1..11} ; do python python/rnn_wings.py ; done
mv report-pyeddl.log logs/report-pyeddl-with-bn.log

for i in {1..11} ; do python python/rnn_wings_keras.py ; done
mv report-keras.log logs/report-keras-with-bn.log
