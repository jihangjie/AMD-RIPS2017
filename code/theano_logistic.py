#!/usr/bin/python

import theano
import theano.tensor as T
import numpy
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

def train(dtype, target, W, b, x, y_):
  x =  T.cast(x,  dtype=dtype)
  y_ = T.cast(y_, dtype=dtype)

  W = theano.shared(value=numpy.asarray(W.eval(), dtype=dtype), name='W', borrow=True)
  b = theano.shared(value=numpy.asarray(b.eval(), dtype=dtype), name='b', borrow=True)

  y = T.nnet.softmax(T.dot(x, W) + b)
  cross_entropy = -T.sum(y_*T.log(y))
  
  index = T.lscalar()  # index to a [mini]batch
  
  if dtype == 'float64':
    learning_rate = numpy.float64(.1)
  if dtype == 'float32':
    learning_rate = numpy.float32(.1)
  if dtype == 'float16':
    learning_rate = numpy.float16(.1)
  
  # compute the gradient of cost with respect to theta = (W,b)
  g_W = T.grad(cost=cross_entropy, wrt=W)
  g_b = T.grad(cost=cross_entropy, wrt=b)

  # specify how to update the parameters of the model as a list of
  # (variable, update expression) pairs.
  updates = [(W, W - learning_rate * g_W), (b, b - learning_rate * g_b)]

  theano.function(inputs=[x, y_], outputs=cross_entropy, updates=updates)

  train_model = theano.function(inputs=[x, y_], outputs=cross_entropy, updates=updates)
  
  # train_model with mini-batch training
  curr_accuracy = 0
  for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    batch_xs = numpy.asarray(batch_xs, dtype=dtype)
    batch_ys = numpy.asarray(batch_ys, dtype=dtype)
  
    train_model(batch_xs, batch_ys)
    truncate_2d(W, 14)
    truncate_1d(b, 14)
    #next_accuracy = find_accuracy(x, y, y_, dtype)
    #if abs(curr_accuracy - next_accuracy) <= target:
    #  print i
    #  break
    #curr_accuracy = next_accuracy
  return W, b, x, y, y_

def find_accuracy(x, y, y_, dtype):
  test_set_x = numpy.asarray(mnist.test.images, dtype=dtype)
  test_set_y = numpy.asarray(mnist.test.labels, dtype=dtype)
  
  correct_prediction = T.eq(T.argmax(y, 1), T.argmax(y_, 1))
  accuracy = T.mean(T.cast(correct_prediction, dtype))
  accuracy_f = theano.function(inputs=[x, y_], outputs=accuracy)
  return accuracy_f(test_set_x, test_set_y)
  

def main():
  dtype0 = 'float32'
  dtype1 = 'float32'
  W = theano.shared(value=numpy.zeros((784, 10), dtype=dtype0), name='W', borrow=True)
  b = theano.shared(value=numpy.zeros((10,), dtype=dtype0), name='b', borrow=True)
  x = T.matrix('x', dtype=dtype0)
  y_ = T.matrix('y', dtype=dtype0)
  
  W, b, x, y, y_ = train(dtype0, .00001, W, b, x, y_)
  #W, b, x, y, y_ = train(dtype1, .00000001, W, b, x, y_)
  print find_accuracy(x, y, y_, dtype0)

if __name__ == "__main__":
  main()
