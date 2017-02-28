"""
 A version of dA for playing around with Theano functions. To begin with,
 try plotting / analysing the distribution on weight values / derivatives
 as learning progresses.


"""

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


try:
    import PIL.Image as Image
except ImportError:
    import Image

class dA(object):

    def __init__(
            self,
            numpy_rng,
            theano_rng = None,
            input=None,
            n_visible=784,
            n_hidden=500,
            W=None,
            bhid=None,
            bvis=None
    ):
        self.n_hidden = n_hidden
        self.n_visible = n_visible

        #create a symbolic random variable:
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(23))

        if not W:
            W_values = numpy.asarray(
                numpy_rng.uniform(
                    low=12 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=16 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=4*numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=4*numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        # we are using tied weights, in which the output weights are just the
        # transpose of the input ones.
        self.W = W
        self.W_prime = W.T
        self.b = bhid
        self.b_prime = bvis

#        deltas = numpy.zeros(shape=(n_visible, n_hidden), dtype=theano.config.floatX)
        self.deltaW = theano.shared(
            value=numpy.zeros(shape=(n_visible, n_hidden), dtype=theano.config.floatX),
            borrow=True
        )

        self.deltaBvis = theano.shared(
            value=numpy.zeros(n_visible, dtype=theano.config.floatX),
            borrow=True
        )

        self.deltaBhid = theano.shared(
            value=numpy.zeros(n_hidden, dtype=theano.config.floatX),
            borrow=True
        )

        self.theano_rng = theano_rng

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        # bundle up all the params. No W_prime as is updated whenever W is updated.
        self.params = [self.W, self.b, self.b_prime]
        self.gparams = []
        self.deltaParams = [self.deltaW, self.deltaBhid, self.deltaBvis]


    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)


    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input


    def get_cost_updates(self, corruption_level, learning_rate):

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)
        self.gparams = T.grad(cost, self.params)

        updates1 = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, self.gparams)
        ]

        updates2 = [
            (deltaParam, -(learning_rate * gparam))
            for deltaParam, gparam in zip(self.deltaParams, self.gparams)
        ]

        updates=updates1+updates2

        return (cost, updates)

    def paramsToArray(self, params):

        x = numpy.array([])
        for par in params:
            temp = numpy.asarray(par.get_value())
            x = numpy.hstack((x, numpy.reshape(temp, temp.size)))
        return x



def test_dA(learning_rate=0.1, training_epochs=15, dataset='mnist.pkl.gz', batch_size=20, output_folder='dA_plots'):

    datasets = load_data(dataset)
    #datasets is a list of Theano tensors
    train_set_x, train_set_y = datasets[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=28 * 28, n_hidden=500)

    cost, updates = da.get_cost_updates(
        corruption_level=0.0,
        learning_rate=learning_rate
    )

    print (len(updates))


    # Build the graph:
    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    for epoch in range(training_epochs):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
            plt.figure(1)
            plt.hist(da.paramsToArray(da.params), 200, normed=1, facecolor='blue')
            plt.figure(2)
            plt.hist(da.paramsToArray(da.deltaParams), 200, normed=1, facecolor='red')
            plt.show()

        print("epoch %d, cost", epoch, numpy.mean(c))




if __name__ == '__main__':
    test_dA(training_epochs=10)
    print('apparent success!')
