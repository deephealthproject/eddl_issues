import os
import sys
import pickle
import numpy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pyeddl import eddl, tensor

from matplotlib import pyplot


f = open('data/data.pckl', 'rb')
data = pickle.load(f)
print(data.keys())
f.close()
X = data['X']
y = data['y']


'''
values = X.flatten().copy()
values = numpy.exp(values)
n, bins, patches = pyplot.hist(values[values < 10], bins = 100, density = True)
pyplot.show()
'''


print(X.shape, X.min(), X.max(), X.ptp(), X.mean(), X.std())
#X = StandardScaler().fit_transform(X.reshape([-1, 1])).reshape(X.shape)

'''
p = numpy.percentile(X.flatten(), numpy.linspace(0, 100, 101), interpolation = 'linear')

print(p)

values = X.flatten().copy()
values.sort()
print(values[:20])
print(values[-20:])
m = values.mean()
s = values.std()
print('ousiders on the left',  sum(values < (m - 3 * s)))
print('ousiders on the right', sum(values > (m + 3 * s)))
'''

print(X.shape, X.min(), X.max(), X.ptp(), X.mean(), X.std())
print(y.shape, y.min(), y.max(), y.ptp())

'''
# Uncomment these lines to generate synthetic samples for
# a sanity check of the models.
# Syntetic data must runs OK
#
for i in range(len(X)):
    if y[i] == 0:
        X[i, :, :] = 10 + numpy.random.randn(X.shape[1], X.shape[2]) * 3
    else:
        X[i, :, :] = -3 + numpy.random.randn(X.shape[1], X.shape[2]) * 10
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

eX_train = tensor.Tensor.fromarray(X_train)
ey_train = tensor.Tensor.fromarray(y_train)
eX_test = tensor.Tensor.fromarray(X_test)
ey_test = tensor.Tensor.fromarray(y_test)

eX_train.info()
ey_train.info()
eX_test.info()
ey_test.info()

eX_train.save('data/X_train.bin', format = 'bin')
ey_train.save('data/y_train.bin', format = 'bin')
eX_test.save('data/X_test.bin', format = 'bin')
ey_test.save('data/y_test.bin', format = 'bin')
