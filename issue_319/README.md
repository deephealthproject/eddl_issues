# ISSUE 319

Here you can find the code (Python and C++) used to check issue 319. Initially, the issue was related to a discrepancy between the inference results obtained
when using EDDL and Keras with the same model (same topology and weights). That part was solved by changing some flags in the installation of the EDDL that were not correct.

But we found some discrepancies when training with the different toolkits. So this could be related with the [issue 302](https://github.com/deephealthproject/eddl/issues/302),
where we can see a difference in the results obtained when training with EDDL and Pytorch when using (supposedly) the same model and training pipeline.

The code in this folder implements an experiment to check how the results when training with different libraries differ, even with the same initialization:

1. Use the CIFAR10 dataset to train a model using Pytorch, Keras and EDDL

2. Save the trained models with each library and the model with just the initialized weights (with each library too)

3. Import the pretrained models with the other libraries and check that the inference result is the same that was obtained with the original library

4. Import the model with just the initialized weights with the other libraries and perform training. Then check which is the difference with the result after
   training with the original library

The folder contains:

- [init_keras_cifar.py](init_keras_cifar.py): The code to train models and perform inference (importing from ONNX) with Keras
- [init_pytorch_cifar.py](init_pytorch_cifar.py): The code to train models and perform inference (importing from ONNX) with Pytorch
- [init_eddl_cifar.cpp](init_eddl_cifar.cpp): The code to train models and perform inference (importing from ONNX) with EDDL
- [plot_results.py](plot_results.py): Script that takes the inference results of two experiments and compares them creating plots
- [plots](plots): Folder with the plots generated with all the experiments
- [outputs](outputs): Folder with the std output generated by the scripts with each experiment
- [results](results): Folder with the CSV files with the test inference results from each experiment used to create the plots