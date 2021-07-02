import os
import sys
import pickle
import numpy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise, BatchNormalization, Input, LSTM, GRU, Flatten, ReLU
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical



f = open('data/data.pckl', 'rb')
data = pickle.load(f)
print(data.keys())
f.close()
X = data['X']
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)


l_input = Input((7, 26))
l = l_input
l = GRU(256, activation = 'linear')(l)
l = BatchNormalization()(l)
l = ReLU()(l)
l = Dense(256, activation = 'linear')(l)
l = BatchNormalization()(l)
l = ReLU()(l)
l_output = Dense(2, activation = 'softmax')(l)

model = Model([l_input], [l_output])
model.summary()


#optimizer = SGD(lr = 1.0e-2)
#optimizer = RMSprop(lr = 1.0e-2)
optimizer = Adam(lr = 1.0e-3)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

model.fit(X_train, Y_train, batch_size = 10, epochs = 30, shuffle = True, verbose = 1, validation_data = (X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose = 0)
print(score)


z_train = model.predict(X_train).argmax(axis = 1)
z_test = model.predict(X_test).argmax(axis = 1)

print(confusion_matrix(y_train, z_train))
print(confusion_matrix(y_test, z_test))
