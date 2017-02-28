import theano as th
from theano import tensor as T
import numpy as np


i = T.iscalar('i') #Number of iterations.
x0 = T.ivector('x0') #Initializes the recurrence, since we need the previous 
                    #two terms in the Fibonacci sequence, we need an initial vector
                    #with two terms.
results, updates = th.scan(fn=lambda f_m_1,f_m_2: f_m_1+f_m_2,
                            outputs_info=[{'initial':x0, 'taps':[-2,-1]}],
                            n_steps=i)




f=th.function(inputs=[i,x0], outputs=results, updates=updates)
print ("*** ", f(30, np.asarray([ 2, 3], dtype=np.int32)))


# implement Newton-Raphson 

x=T.dscalar('x')
f = 6*x**3 - 2*x**2 +9*x+ 1 + T.cos(x)
f_prime = T.grad(f,x)
f_prime_2 = T.grad(f_prime, x)

#Then the compiled theano functions for plotting.
F = th.function(inputs=[x], outputs=f)
F_prime = th.function(inputs=[x], outputs=f_prime)
F_prime_2 = th.function(inputs=[x], outputs=f_prime_2)

#Now let's make a plot.
xs = np.linspace(-1,1,1000)
y1 = [F(z) for z in xs]
y2 = [F_prime(z) for z in xs]
y3 = [F_prime_2(z) for z in xs]

import matplotlib.pyplot as plt
#%matplotlib inline
plt.plot(xs, y1, label='f')
plt.plot(xs, y2, label='f\'')
plt.plot(xs, y3, label='f\'\'')
plt.plot(xs, [0 for z in xs], label='zero')
plt.legend()

#plt.show()

i=T.iscalar('k')


# The reason we're using this fancy nested function arrangement is because we want to be able to 
# swap out the actual function, but still keep the solving code unchanged. 

def update_func(func):
    #Argument func is a function producing the symbolic variable representation of the function we want to zero.
    def update(z):
        return z-func(z)/T.grad(func(z),z)
    return update

def f(z):
    return  6*z**3 - 2*z**2 +9*z+ 1 + T.cos(z)

def g(z):
    return  3*z**3 - 5*z**2 +9*z+ 1 + T.cos(z)


results, updates = th.scan(fn=update_func(g),
                           outputs_info = x,
                           n_steps=i)

NR = th.function(inputs=[x,i], outputs=results[-1], updates=updates)

print NR(0.21, 30)


x=T.vector('x')
outputs_info = np.float64(0)

results, updates = th.scan(fn=lambda x_m_1, x0: x_m_1+x0,
                           outputs_info=outputs_info,
                           sequences=x)


X=np.asarray([1,1,3,5,11,-9], dtype=np.float32)
print sum(X)

print results[-1].eval({x:X})

mysum = th.function([x], results, updates=updates)
print(mysum(X)[-1])



# Conditionals

from theano.ifelse import ifelse
import time


a,b = T.scalars('a', 'b')
x,y = T.matrices('x','y')


z_switch = T.switch(T.lt(a,b), T.mean(x), T.mean(y))
z_lazy = ifelse(T.lt(a, b), T.mean(x), T.mean(y))

f_switch = th.function([a, b, x, y], z_switch, 
mode=th.Mode(linker='vm')) # allows lazy evaluation when used with ifelse

f_lazy = th.function([a, b, x, y], z_lazy, 
mode=th.Mode(linker='vm')) # allows lazy evaluation when used with ifelse


x1 = np.random.randn(20)
x2 = np.random.randn(20)

val1 = 0.
val2 = 1.

big_mat1 = np.random.randn(10000, 1000)
big_mat2 = np.random.randn(10000, 1000)

print("mat1: ", np.mean(big_mat1))
print("mat2: ", np.mean(big_mat2))

n_times = 10

tic = time.clock()
for i in range(n_times):
    print(f_switch(x1, x2, big_mat1, big_mat2))
print('time spent evaluating both values %f sec' % (time.clock() - tic))

tic = time.clock()
for i in range(n_times):
    f_lazy(val1, val2, big_mat1, big_mat2)
print('time spent evaluating one value %f sec' % (time.clock() - tic))

