
from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images
import matplotlib.pyplot as plt

from myDa import dA


dataset='mnist.pkl.gz'

learning_rate=0.1
training_epochs=1
batch_size=20

#load the data
datasets = load_data(dataset)
# datasets is a list of Theano tensors
train_set_x, train_set_y = datasets[0]
n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data is presented as rasterized images

rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=28 * 28, n_hidden=500)

cost, updates = da.get_cost_updates(
    corruption_level=0.0,
    learning_rate=learning_rate
)

# Build the graph:
train_da = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size]
    }
)

n_visible=784
n_hidden=500
deltas = numpy.zeros(shape=(n_visible, n_hidden), dtype=theano.config.floatX)
D = theano.shared(value=deltas, name='D', borrow=True)

for epoch in range(training_epochs):
    c = []
    plt.figure(1)
    plt.hist(da.paramsToArray(da.params), 200, normed=1, facecolor='blue')
#    plt.hist(da.paramsToArray(da.gparams), 200, normed=1, facecolor='red')
    plt.show()
    for batch_index in range(n_train_batches):
        c.append(train_da(batch_index))

    print("epoch %d, cost", epoch, numpy.mean(c))


