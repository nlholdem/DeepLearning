import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np

# define network inputs and output

x = T.dvector()
y = T.dscalar()

# Compute output of a layer and vector of inputs
# Add a bias term
def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x) #theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
    h = nnet.sigmoid(m)
    return h

# gradient descent - returns new weight values so we can use in an update expression
def grad_desc(cost, theta):
    alpha = 0.1 # learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))

# create the weights

# input to hidden
theta1 = theano.shared(np.array(np.random.rand(3,3), dtype=theano.config.floatX)) # randomly initialize
# hidden to single output node
theta2 = theano.shared(np.array(np.random.rand(4,1), dtype=theano.config.floatX))

# calculate hidden layer activation using layer()
hid1 = layer(x, theta1)

# calculate outputs. We use T.sum to give a scalar from a matrix
out1 = T.sum(layer(hid1, theta2)) #output layer
fc = (out1 - y)**2 #cost expression

# grad_desc above requires cost to be a theano variable computed from a graph. So let's define it
# note that this is also updating the weights!
cost = theano.function(inputs=[x, y], outputs=fc, updates=[
        (theta1, grad_desc(fc, theta1)),
        (theta2, grad_desc(fc, theta2))])

# create a separate function to run the network, without doing any learning:
run_forward = theano.function(inputs=[x], outputs=out1)


# train the thing
inputs = np.array([[0,1],[1,0],[1,1],[0,0]]).reshape(4,2) #training data X
exp_y = np.array([0.9, 0.9, 0.1, 0.1]) #training data Y
cur_cost = 0
for i in range(10000):
    for k in range(len(inputs)):
        cur_cost = cost(inputs[k], exp_y[k]) #call our Theano-compiled cost function, it will auto update weights
    if i % 500 == 0: #only print the cost every 500 epochs/iterations (to save space)
        print('Cost: %s' % (cur_cost,))

# test it
print(run_forward([0,1]))
print(run_forward([1,1]))
print(run_forward([1,0]))
print(run_forward([0,0]))


