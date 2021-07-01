'''
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2021, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: July 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*
'''

#/////////////////////////////////
#/ Testing issue generated by George Mavrakis from WINGS
#/ Using fit for training
#/////////////////////////////////

import os
import sys
import time
import numpy
import pickle
from pyeddl import eddl, tensor

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == '__main__':
    # Default settings
    epochs = 10
    batch_size = 10
    use_cpu = False
    output_softmax = True

    for i in range(len(sys.argv)):
        if sys.argv[i] == "--cpu":
            use_cpu = True
        elif sys.argv[i] == "--epochs":
            epochs = int(sys.argv[i + 1])
        elif sys.argv[i] == "--batch-size":
            batch_size = int(sys.argv[i + 1])
        elif sys.argv[i] == "--softmax-at-output":
            output_softmax = True
        elif sys.argv[i] == "--sigmoid-at-output":
            output_softmax = False


    # Settings
    num_classes = 2

    # Define network
    input_layer = eddl.Input([26])
    _layer_ = input_layer

    _layer_ = eddl.GRU(_layer_, 256)
    #_layer_ = eddl.LSTM(_layer_, 256)
    _layer_ = eddl.BatchNormalization(_layer_, True)
    _layer_ = eddl.ReLu(_layer_)
    _layer_ = eddl.Dense(_layer_, 256)
    _layer_ = eddl.BatchNormalization(_layer_, True)
    _layer_ = eddl.ReLu(_layer_)

    if output_softmax:
        _layer_ = eddl.Dense(_layer_, num_classes)
        output_layer = eddl.Softmax(_layer_)
    else:
        _layer_ = eddl.Dense(_layer_, 1)
        output_layer = eddl.Sigmoid(_layer_)

    net = eddl.Model([input_layer], [output_layer]);

    if use_cpu:
        cs = eddl.CS_CPU(4)
    else:
        cs = eddl.CS_GPU([1], mem = "full_mem")

    #optimizer = eddl.sgd(1.0e-3)
    #optimizer = eddl.rmspropo(1.0e-3)
    optimizer = eddl.adam(1.0e-2)

    losses = ["softmax_cross_entropy"] if output_softmax else ["mse"]
    metrics = ["categorical_accuracy"] if output_softmax else ["binary_accuracy"]

    # Build model
    eddl.build(net, optimizer, losses, metrics, cs = cs, init_weights = True)

    # View model
    eddl.summary(net);


    # Load data
    '''
    X_train = tensor.Tensor.load("data/X_train.bin", format = 'bin')
    y_train = tensor.Tensor.load("data/y_train.bin", format = 'bin')
    X_test  = tensor.Tensor.load("data/X_test.bin",  format = 'bin')
    y_test  = tensor.Tensor.load("data/y_test.bin",  format = 'bin')

    X_train_np = X_train.getdata()
    y_train_np = y_train.getdata()
    X_test_np  = X_test.getdata()
    y_test_np  = y_test.getdata()
    '''
    f = open('data/data.pckl', 'rb')
    data = pickle.load(f)
    print(data.keys())
    f.close()
    X = data['X']
    y = data['y']

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size = 0.20, random_state = 0)
    #Y_train = to_categorical(y_train_np)
    #Y_test = to_categorical(y_test_np)


    Y_train_np = numpy.zeros([len(y_train_np), 1, 2 if output_softmax else 1])
    z_train_np = numpy.zeros([len(y_train_np), 1, 1])
    Y_test_np  = numpy.zeros([len(y_test_np),  1, 2 if output_softmax else 1])
    z_test_np  = numpy.zeros([len(y_test_np),  1, 1])

    #print(Y_train_np.shape)
    #print(y_train_np.shape)

    if output_softmax:
        Y_train_np[y_train_np[:, 0] == 0, 0, 0] = 1
        Y_train_np[y_train_np[:, 0] == 1, 0, 1] = 1
        Y_test_np[y_test_np[:, 0] == 0, 0, 0] = 1
        Y_test_np[y_test_np[:, 0] == 1, 0, 1] = 1
    else:
        Y_train_np[:, 0, :] = y_train_np[:, :]
        Y_test_np[:, 0, :]  = y_test_np[:, :]


    #print("min(X_train) = %.6f   max(X_train) = %.6f\n" % (X_train_np.min(), X_train_np.max()))

    # Preprocessing
    mean = X_train_np.mean()
    std = X_train_np.std()
    X_train_np = (X_train_np - mean) / std
    X_test_np = (X_test_np - mean) / std

    batches_x_epoch = len(X_train_np) // batch_size
    # Train model
    for epoch in range(epochs):
        '''
        fit(net, {X_train}, {Y_train}, batch_size, 1);
        evaluate(net, {X_test}, {Y_test}, batch_size);
        '''
        eddl.reset_loss(net)
        for j in range(len(X_train_np) // batch_size):

            xbatch = tensor.Tensor.fromarray(X_train_np[j * batch_size : (j + 1) * batch_size])
            ybatch = tensor.Tensor.fromarray(Y_train_np[j * batch_size : (j + 1) * batch_size])

            eddl.train_batch(net, [xbatch], [ybatch])

            #print(j + 1,  'loss:', eddl.get_losses(net), 'metrics:', eddl.get_metrics(net), end = '                                          \r', flush = True)
            print(j + 1,  'loss:', eddl.get_losses(net), end = '                                          \r', flush = True)
        print()
        eddl.evaluate(net, [tensor.Tensor.fromarray(X_train_np)], [tensor.Tensor.fromarray(Y_train_np)], bs = 10)
        eddl.evaluate(net, [tensor.Tensor.fromarray(X_test_np)],  [tensor.Tensor.fromarray(Y_test_np)], bs = 10)
    #
    #

    cm_list = dict()
    for subset in [('train', X_train_np, y_train_np), ('test', X_test_np, y_test_np)]:
        name, X, y_true = subset
        y_true = y_true.flatten().astype(int)
        y_pred = list()
        for i in range(0, len(X), batch_size):
            xbatch = tensor.Tensor.fromarray(X[i : i + batch_size])
            z = eddl.predict(net, [xbatch])[0] # there is only one output layer, it is at position 0
            z = z.getdata() # easier to work with NumPy arrays
            if z.shape[1] == 2:
                y_pred = y_pred + [_ for _ in z.argmax(axis = 1)]
            else:
                y_pred = y_pred + [1 if _ else 0 for _ in z[:, 0] > 0.5]
        #
        y_pred = numpy.array(y_pred).astype(int)
        #
        cm_list[name] = (y_true, y_pred)
    #

    log_file = open('report.log', 'at')
    log_file.write(f'\nRUN at {time.asctime()} \n')
    log_file.write('    using ')
    log_file.write('softmax' if output_softmax else 'sigmoid')
    log_file.write(' as the activation for the output layer\n')
    log_file.write(f'    trained during {epochs} epochs\n')
    for key, (y_true, y_pred) in cm_list.items():
        log_file.write('\n')
        log_file.write(f'Confusion matrix for subset {key}:\n')
        log_file.write(str(confusion_matrix(y_true, y_pred, labels = [0, 1])))
        log_file.write('\n\n')
        log_file.write(f'Classification  report for subset {key}:\n')
        log_file.write(classification_report(y_true, y_pred))
        log_file.write('\n\n')
    log_file.close()

    os.system("tail -n 36 report.log")
