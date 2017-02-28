import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np

# define network inputs and output

x = T.dscalar()
fx = T.exp(T.sin(x**2))

f = theano.function(inputs=[x], outputs=[fx])

fp = T.grad(fx, wrt=x)
fprime = theano.function([x], fp)

fprime(15)

