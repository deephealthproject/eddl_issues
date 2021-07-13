# ISSUE 294

Different results of recurrent GRU network during different executions [Issue 294](https://github.com/deephealthproject/eddl/issues/294)

Here you can find C++ and Python code used to check the issue.
We were not able to reproduce the behaviour reported.
The files [report-cpp.log](report-cpp.log) and [report-python.log](report-python.log)
contain the result of different runs with the C++ and Python code.

*C++* version was checked with the develop branch at 2020-07-01.
*Python* version was checked with _pyeddl_ and _eddl_ version 1.0


This folder also contains:

- [here](data) the sample data provided by Wings to help us solve [*Issue 294*](https://github.com/deephealthproject/eddl/issues/294),

- [here](python) a sample Python code with the same basic topology tested with _TensorFlow+Keras_ and to extract the sample data to the binary tensor format of the *EDDL*, and 

- [here](scripts) some useful scripts to run the compile C++ code.


Findings:

- Discrepancies using GRU between *pyeddl-cuDNN* and *Keras*:
    - **Solved**: similar behaviour using both toolkits with the same initial network weights

- Differences when using same initial weights in several runs:
    - CPU using optimizer SGD: no differences **Solved**
    - CPU using optimizer Adam: insignificant differences **Solved**
    - CPU using optimzier SGD and loading initial weights from ONNX:  insignificant differences **Solved**
    - CPU using optimzier Adam and loading initial weights from ONNX:  insignificant differences **Solved**

    - GPU using optimizer SGD: differences to be studied **Pending**
    - GPU using optimizer Adam: differences to be studied **Pending**
    - GPU using optimzier SGD and loading initial weights from ONNX:  differences to be studied **Pending**
    - GPU using optimzier Adam and loading initial weights from ONNX:  differences to be studied **Pending**


- Fixed bug in BatchNormalization:
    - to explain in the code
    - to review which version of forward and backward to use for GPU with CUDA. CPU and cuDNN are clear


- BatchNormalization ONNX import/export:
    - to review the 'n-parameters', only shape[1] is used
    - do we have to accept values for momentum not in the range [0.9, 0.9999] ?
