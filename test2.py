import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
from theano import pp 

# some simple functions and automatic derivatives; vectors and scalars

x = T.dvector()
m = T.dmatrix()
y = T.sum(x **2)
y1 = T.sum(m ** 2)
dydx = T.grad(y, wrt=x)
dydm = T.grad(y1, wrt=m)

func = theano.function(inputs=[x], outputs=y)
dfunc = theano.function(inputs=[x], outputs=dydx)

func1 = theano.function(inputs=[m], outputs=y1)
dfunc1 = theano.function(inputs=[m], outputs=dydm)


# let's try and build a Jacobian
# first work out how to use scan:




pp(dydx)

print(func([1,2]))
print(dfunc([1,2]))

print(func([2]))
print(dfunc([2]))

print(func([3]))
print(dfunc([3]))

print(func([4]))
print(dfunc([4]))

print(func([5]))
print(dfunc([5]))

print(func1([[1,2],[2,3]]))
print(dfunc1([[1,2],[2,3]]))

