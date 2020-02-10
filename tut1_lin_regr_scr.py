from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)
data_ctx = mx.cpu()
model_ctx = mx.cpu()
num_inputs = 2
num_outputs = 1
num_examples = 10000

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

X = nd.random_normal(shape=(num_examples, num_inputs), ctx=data_ctx)
noise = .1 * nd.random_normal(shape=(num_examples,), ctx=data_ctx)
y = real_fn(X) + noise

print(X[0])
print(y[0])


print(2 * X[0, 0] - 3.4 * X[0, 1] + 4.2)


plt.scatter(X[:, 1].asnumpy(),y.asnumpy())
plt.show()

batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                   batch_size=batch_size, shuffle=True)

for i, (data, label) in enumerate(train_data):
    print(data, label)
    break


for i, (data, label) in enumerate(train_data):
    print(data, label)
    break

