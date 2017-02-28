

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data

import numpy
import matplotlib.pyplot as plt


n_in = 20
n_out = 14

rng = numpy.random.RandomState(89677)

W_values = numpy.asarray(
    rng.uniform(
        low=-numpy.sqrt(6. / (n_in + n_out)),
        high=numpy.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)
    ),
    dtype=theano.config.floatX
)


def myGetUpdates(state, inc):
    print ("** myGetUpdates **")
    return [(state, state+inc)]

state = theano.shared(0)
inc = T.iscalar('inc')

updates = myGetUpdates(state, inc)
acc = theano.function([inc], state, updates=updates)
print (state.get_value())

acc(100)
print (state.get_value())
acc(3)
acc(35)
print (state.get_value())


#plt.hist(numpy.reshape(W_values, W_values.size), 50, normed=1, facecolor='blue')
#plt.show()

W = theano.shared(value=W_values, name='W', borrow=True)

WMat = [W,W]
WMatMix = [W, W.T]
