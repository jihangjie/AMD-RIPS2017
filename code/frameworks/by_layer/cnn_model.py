import theano
import theano.tensor as T
import numpy as np

from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d

def floatX(X, dtype):
  return np.asarray(X, dtype=dtype)

def init_weights(shape, dtype, perturbation = 0.):
  np.random.seed(42)
  return theano.shared(floatX(np.random.randn(*shape) * .01 + perturbation, dtype))

def rectify(x):
  return T.maximum(x, 0.)

def first_layer(x, w_c1, b_c1):
  c1 = rectify(conv2d(x, w_c1) + b_c1.dimshuffle('x', 0, 'x', 'x'))
  p1 = pool_2d(c1, (3, 3), ignore_border = False)
  return p1

def second_layer(p1, w_c2, b_c2):
  c2 = rectify(conv2d(p1, w_c2) + b_c2.dimshuffle('x', 0, 'x', 'x'))
  p2 = pool_2d(c2, (2, 2), ignore_border = False)
  return p2

def dense_layer(p2, w_h3, b_h3):
  p2_flat = p2.flatten(2)
  h3 = rectify(T.dot(p2_flat, w_h3) + b_h3)
  return h3

def output_layer(h3, w_o, b_o):
  p_y_given_x = T.nnet.softmax(T.dot(h3, w_o) + b_o)
  return p_y_given_x
