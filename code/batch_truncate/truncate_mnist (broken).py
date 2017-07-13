#!/usr/bin/python

import theano
import theano.tensor as T
import numpy as np
import truncate

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def truncate_2n(x, bitsize=32):
  ''' truncate numpy array
      @param x: numpy array
      @return a truncated numpy array
  '''
  value = x.eval()
  for row in range(value.shape[0]):
    for col in range(value.shape[1]):
      value[row][col] = truncate.truncate(value[row][col], bitsize)
  x.assign(value)
  return x

def truncate_1n(x, bitsize=32):
  ''' truncate numpy array
      @param x: numpy array
      @return a truncated numpy array
  '''
  value = x.eval()
  for num in range(value.shape[0]):
      value[num] = truncate.truncate(value[num], bitsize)
  x.assign(value)
  return x

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
  
# train_model with mini-batch training
def train(dtype, W, b, numPrecision):
  W = theano.shared(value=np.asarray(W.eval(), dtype=dtype), name='W', borrow=True)
  b = theano.shared(value=np.asarray(b.eval(), dtype=dtype), name='b', borrow=True)
  for i in range(100):
    x, y_ = mnist.train.next_batch(100)

    x = np.asarray(x, dtype=dtype)
    y_ = np.asarray(y_, dtype=dtype)
    mul = theano.shared(value=np.asarray(T.dot(x, W).eval(), dtype=dtype), name='mul', borrow=True)
    mul = truncate_2d(mul, numPrecision)
    suMul = theano.shared(value=np.asarray((mul+b).eval(), dtype=dtype), name='suMul', borrow=True)
    suMul = truncate_2d(suMul, numPrecision)
    
    y = T.nnet.softmax(truncate_2n(T.dot(x, W)+b))
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
    print(g_W.eval())
    print(g_b.eval())
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    W1 = W1 - learning_rate * g_W.eval()
    b1 = b1 - learning_rate * g_b.eval()
    W1 = truncate_2n(W1, numPrecision)
    b1 = truncate_1n(b1, numPrecision)
  return W1, b1

#evaluate accuracy of the model using 1000 randomly selected samples from test data
def find_accuracy(W, b, dtype):
  test_set_x = np.asarray(mnist.test.images, dtype=dtype)
  test_set_y = np.asarray(mnist.test.labels, dtype=dtype)
  inx = np.random.randint(len(test_set_x), size = 1000)
  correct_prediction = T.eq(T.argmax(T.nnet.softmax(T.dot(test_set_x[inx], W) + b), 1), T.argmax(test_set_y[inx], 1))
  accuracy = T.mean(T.cast(correct_prediction, dtype))
  return accuracy.eval()
  

def main():
  dtype0 = 'float32'
  numBits = 32
  #while numBits > 8:
  W = theano.shared(value=np.random.random_sample((784, 10)), name='W', borrow=True)
  b = theano.shared(value=np.random.random_sample((10,)), name='b', borrow=True)
  
  W, b = train(dtype0, W, b, numBits)
  print find_accuracy(W, b, dtype0)
  #numBits = numBits-1

if __name__ == "__main__":
  main()
