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

def dense_layer(x, w_h, b_h):
  x_flat = x.flatten(2)
  h = rectify(T.dot(x_flat, w_h) + b_h)
  return h

def output_layer(h, w_o, b_o):
  p_y_given_x = T.nnet.softmax(T.dot(h, w_o) + b_o)
  return p_y_given_x
