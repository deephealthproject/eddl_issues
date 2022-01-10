import numpy as np
import pandas as pd
import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, ReLU
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import onnx
import tf2onnx
from onnx2keras import onnx_to_keras

# Training settings
parser = argparse.ArgumentParser(description='Keras Conv2D CIFAR10 Example')
parser.add_argument('--onnx-file', type=str, default="", metavar='filepath',
                    help='Path to the ONNX file to use as initialization')
parser.add_argument('--onnx-input-name', type=str, default="", metavar='name',
                    help='Name of the input layer of the model provided')
parser.add_argument('--channels-first', action='store_true', default=False,
                    help='Changes data from HWC to CHW')
parser.add_argument('--inference', action='store_true', default=False,
                    help='Avoid the training phase')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--output-path', type=str,
                    default="keras_conv2D_cifar.onnx",
                    help='Output path to store the onnx file')
args = parser.parse_args()

# Load CIFAR10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (32, 32, 3)
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

if args.channels_first:
    x_train = tf.transpose(x_train, [0, 3, 1, 2])
    x_test = tf.transpose(x_test, [0, 3, 1, 2])

# Get one hot encoding from labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Train data shape:", x_train.shape)
print("Train labels shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test labels shape:", y_test.shape)

if not args.onnx_file:
    model = Sequential()
    model.add(Input(shape=(32, 32, 3), name="linput"))
    model.add(Conv2D(16, 3))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, 3))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, 3))
    model.add(ReLU())
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
else:
    onnx_model = onnx.load(args.onnx_file)
    assert args.onnx_input_name != "", "Provide the input name of the ONNX"
    model = onnx_to_keras(onnx_model, [args.onnx_input_name], change_ordering=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(args.lr),
              metrics=['accuracy'])

model.summary()

# Convert to ONNX
if args.channels_first:
    input_spec = (tf.TensorSpec((args.batch_size, 3, 32, 32), dtype=tf.float32),)
else:
    input_spec = (tf.TensorSpec((args.batch_size, 32, 32, 3), dtype=tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec)
# Save ONNX to file
onnx.save(onnx_model, f'before_fit_{args.output_path}')

# Training
if not args.inference:
    model.fit(x_train,
              y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              shuffle=False)

# Evaluation
res = model.evaluate(x_test, y_test)
print("Evaluation result: Loss:", res[0], " Accuracy:", res[1])

# Predict to obtain the values for the plots
outs = model.predict(x_test)
preds = np.argmax(outs, axis=1)
pred_confs = np.max(outs, axis=1)
trues = np.argmax(y_test, axis=1)
true_confs = [outs[i, true_class] for i, true_class in enumerate(trues)]
df = pd.DataFrame({'true': trues,
                   'pred': preds,
                   'true_confidence': true_confs,
                   'pred_confidence': pred_confs})
df.to_csv("keras_results.csv", index=False)

# Convert to ONNX
if args.channels_first:
    input_spec = (tf.TensorSpec((args.batch_size, 3, 32, 32), dtype=tf.float32),)
else:
    input_spec = (tf.TensorSpec((args.batch_size, 32, 32, 3), dtype=tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec)
# Save ONNX to file
onnx.save(onnx_model, args.output_path)
