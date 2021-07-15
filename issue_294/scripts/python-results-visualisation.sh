#!/bin/bash

options=" --to-png"

python python/results-visualisation.py ${options}  --title "Keras: softmax gpu-cuDNN adam 50 epochs"   logs/report-keras-with-bn.log
python python/results-visualisation.py ${options}  --title "pyeddl: softmax gpu-cuDNN adam 50 epochs"  logs/report-pyeddl-with-bn.log
