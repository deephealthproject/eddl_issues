/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: 0.9
 * copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research
 * Centre Date: November 2020 Author: PRHLT Research Centre, UPV,
 * (rparedes@prhlt.upv.es), (jon@prhlt.upv.es) All rights reserved
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace eddl;

////////////////////3/////////////
// test_onnx_conv2D_cifar.cpp:
// A CIFAR10 example with conv2D net
// to test ONNX module
//////////////////////////////////

int main(int argc, char **argv) {
  // Settings
  int epochs = 200;
  int batch_size = 100;
  int num_classes = 10;

  // Script flags
  bool use_cpu = false;
  bool fit_ckpt = false;
  bool only_inference = false;
  bool ch_last = false; // For keras models
  string onnx_model_path("");
  string onnx_export_path("eddl_conv2D_cifar.onnx");
  // Process provided args
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--cpu") == 0) {
      use_cpu = true;
      epochs = 1;
    } else if (strcmp(argv[i], "--onnx-file") == 0) {
      fit_ckpt = true;
      onnx_model_path = argv[i + 1];
      ++i; // Skip model file path for next iteration
    } else if (strcmp(argv[i], "--export-file") == 0) {
      onnx_export_path = argv[i + 1];
      ++i; // Skip model file path for next iteration
    } else if (strcmp(argv[i], "--inference") == 0) {
      only_inference = true;
    } else if (strcmp(argv[i], "--channels-last") == 0) {
      ch_last = true;
    } else if (strcmp(argv[i], "--batch-size") == 0) {
      batch_size = atoi(argv[i + 1]);
      ++i; // Skip batch size for next iteration
    } else if (strcmp(argv[i], "--epochs") == 0) {
      epochs = atoi(argv[i + 1]);
      ++i; // Skip epochs for next iteration
    }
  }

  // Download cifar
  download_cifar10();

  // Load dataset
  Tensor *x_train = Tensor::load("cifar_trX.bin");
  Tensor *y_train = Tensor::load("cifar_trY.bin");
  Tensor *x_test = Tensor::load("cifar_tsX.bin");
  Tensor *y_test = Tensor::load("cifar_tsY.bin");

  if (ch_last) {
    x_train->permute_({0, 2, 3, 1});
    x_test->permute_({0, 2, 3, 1});
  }

  // Preprocessing
  x_train->div_(255.0f);
  x_test->div_(255.0f);

  model net;
  if (!fit_ckpt) {
    // Define network
    layer in = Input({3, 32, 32});
    layer l = in; // Aux var

    l = ReLu(Conv2D(l, 16, {3, 3}, {1, 1}, "valid"));
    l = MaxPool2D(l, {2, 2});
    l = ReLu(Conv2D(l, 32, {3, 3}, {1, 1}, "valid"));
    l = MaxPool2D(l, {2, 2});
    l = ReLu(Conv2D(l, 64, {3, 3}, {1, 1}, "valid"));
    l = GlobalAveragePool2D(l);
    l = Flatten(l);
    layer out = Softmax(Dense(l, num_classes));

    net = Model({in}, {out});

  } else {
    // Import the model ckpt from ONNX
    net = import_net_from_onnx_file(onnx_model_path);
  }

  compserv cs = nullptr;
  if (use_cpu)
    cs = CS_CPU();
  else
    cs = CS_GPU({1});

  // Build model
  build(net,
        adam(0.001),               // Optimizer
        {"softmax_cross_entropy"}, // Losses
        {"categorical_accuracy"},  // Metrics
        cs,                        // Computing Service
        !fit_ckpt);                // Initialize weights

  // View model
  summary(net);

  // Export the untrained model to ONNX
  save_net_to_onnx_file(net, "before_fit_" + onnx_export_path);

  // Train model
  if (!only_inference) {
    // Get the batches manually to get the batches in the same order allways
    int n_batches = x_train->shape[0] / batch_size;
    cout << "Going to train:" << endl;
    for (int e = 1; e <= epochs; ++e) {
      reset_loss(net);
      cout << "Epoch " << e << "/" << epochs << " (" << n_batches << " batches)" << endl;
      int counter = 0;
      for (int b = 0; b <= x_train->shape[0] - batch_size; b += batch_size, ++counter) {
        string batch_indexes = to_string(b) + ":" + to_string(b + batch_size);
        Tensor *x_batch = x_train->select({batch_indexes, ":", ":", ":"});
        Tensor *y_batch = y_train->select({batch_indexes, ":"});

        train_batch(net, {x_batch}, {y_batch});

        print_loss(net, counter);
        cout << "\r";
      }
      cout << endl;
    }
    cout << endl;
  }

  // Evaluate
  evaluate(net, {x_test}, {y_test}, batch_size);

  vtensor vpred_confs;
  vtensor vpreds;
  vector<float> all_true_confs;
  int n_batches = x_test->shape[0] / batch_size;
  int counter = 1;
  cout << endl << "Going to predict with test data to store the results" << endl;
  for (int b = 0; b <= x_test->shape[0] - batch_size; b += batch_size, ++counter) {
    string batch_indexes = to_string(b) + ":" + to_string(b + batch_size);
    Tensor *batch = x_test->select({batch_indexes, ":", ":", ":"});
    Tensor *targets = y_test->select({batch_indexes, ":"});
    Tensor *outs = predict(net, {batch})[0];
    Tensor *preds = outs->argmax({1}, false);
    Tensor *confs = outs->max({1}, false);
    vpreds.push_back(preds);
    vpred_confs.push_back(confs);
    // Get the confidence of the true class
    Tensor *true_labels = targets->argmax({1}, false);
    for (int i = 0; i < batch_size; ++i)
      all_true_confs.push_back(outs->ptr[i*10 + (int)true_labels->ptr[i]]);
  }

  // Gather all the data to create the CSV with the results
  Tensor *all_pred_confs = Tensor::concat(vpred_confs);
  Tensor *all_preds = Tensor::concat(vpreds);
  Tensor *all_trues = y_test->argmax({1}, false);
  ofstream ofile;
  ofile.open("eddl_results.csv");
  ofile << "true,pred,true_confidence,pred_confidence" << endl;
  for (int i = 0; i < y_test->shape[0]; ++i)
    ofile << (int)all_trues->ptr[i] << ","
          << (int)all_preds->ptr[i] << ","
          << all_true_confs[i] << ","
          << all_pred_confs->ptr[i] << endl;
  ofile.close();
  cout << "Predictions stored in eddl_results.csv" << endl;

  // Export the model to ONNX
  save_net_to_onnx_file(net, onnx_export_path);

  delete x_train;
  delete y_train;
  delete x_test;
  delete y_test;
  delete net;

  return EXIT_SUCCESS;
}
