#!/usr/bin/python

import theano
import theano.tensor as T
import numpy as np
import truncate

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def truncate_2d(x, bitsize=32):
  ''' truncate theano varibale
      @param x: theano var
      @return theano variable with truncated value
  '''
  value = x.eval()
  for row in range(value.shape[0]):
    for col in range(value.shape[1]):
      value[row][col] = truncate.truncate(value[row][col], bitsize)
  x.set_value(value)
  return x

def truncate_1d(x, bitsize=32):
  ''' truncate theano varibale
      @param x: theano var
      @return theano variable with truncated value
  '''
  value = x.eval()
  for num in range(value.shape[0]):
      value[num] = truncate.truncate(value[num], bitsize)
  return x

def train(dtype, W, b, numPrecision):
  x = T.matrix('x', dtype=dtype)
  y_ = T.matrix('y', dtype=dtype)
  W = theano.shared(value=np.asarray(W.eval(), dtype=dtype), name='W', borrow=True)
  b = theano.shared(value=np.asarray(b.eval(), dtype=dtype), name='b', borrow=True)

  y = T.nnet.softmax(T.dot(x, W) + b)
  cross_entropy = -T.sum(y_*T.log(y))
  
  if dtype == 'float64':
    learning_rate = np.float64(.1)
  if dtype == 'float32':
    learning_rate = np.float32(.1)
  if dtype == 'float16':
    learning_rate = np.float16(.1)
  
  # compute the gradient of cost with respect to theta = (W,b)
  g_W = T.grad(cost=cross_entropy, wrt=W)
  g_b = T.grad(cost=cross_entropy, wrt=b)

  # specify how to update the parameters of the model as a list of
  # (variable, update expression) pairs.
  updates = [(W, W - learning_rate * g_W), (b, b - learning_rate * g_b)]

  theano.function(inputs=[x, y_], outputs=cross_entropy, updates=updates)

  train_model = theano.function(inputs=[x, y_], outputs=cross_entropy, updates=updates)
  
  # train_model with mini-batch training
  train_set_x = np.asarray(mnist.train.images, dtype=dtype)
  train_set_y = np.asarray(mnist.train.labels, dtype=dtype)
  for start in range(0, len(train_set_x)-129, 128):
    batch_xs = train_set_x[start:start+127]
    batch_ys = train_set_y[start:start+127]
  
    train_model(batch_xs, batch_ys)
    W = truncate_2d(W, numPrecision)
    b = truncate_1d(b, numPrecision)
  return W, b

#evaluate accuracy of the model using 1000 randomly selected samples from test data
def find_accuracy(W, b, dtype):
  test_set_x = np.asarray(mnist.test.images, dtype=dtype)
  test_set_y = np.asarray(mnist.test.labels, dtype=dtype)
  correct_prediction = T.eq(T.argmax(T.nnet.softmax(T.dot(test_set_x, W) + b), 1), T.argmax(test_set_y, 1))
  accuracy = T.mean(T.cast(correct_prediction, dtype))
  return accuracy.eval()
  

def main():
  dtype0 = 'float32'
  numBits = 32
  while numBits > 8:
      W = theano.shared(value=np.zeros((784, 10), dtype=dtype0), name='W', borrow=True)
      b = theano.shared(value=np.zeros((10,), dtype=dtype0), name='b', borrow=True)
  
      W, b = train(dtype0, W, b, numBits)
      print find_accuracy(W, b, dtype0)
      numBits = numBits-1

if __name__ == "__main__":
  main()
