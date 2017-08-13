import theano
import theano.tensor as T
import numpy as np
import math

from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d

from functools import reduce
from operator import mul

def floatX(X, dtype):
  return np.asarray(X, dtype=dtype)

def prod(iterable):
  return reduce(mul, iterable, 1)

def init_weights(shape, dtype):
  np.random.seed(42)
  # calibrating variances with 1/sqrt(n)
  return theano.shared(floatX(np.random.randn(*shape) * math.sqrt(2 / prod(shape)), dtype))
  #return theano.shared(floatX(np.random.randn(*shape) * .01, dtype))

def init_biases(shape, dtype):
  np.random.seed(42)
  return theano.shared(floatX(np.random.randn(*shape) * math.sqrt(2 / prod(shape)), dtype))

def cast_4(trX, trY, X, Y, dtype):
  trX = trX.astype(dtype)
  trY = trY.astype(dtype)
  X = T.cast(X, dtype=dtype)
  Y = T.cast(Y, dtype=dtype)
  return trX, trY, X, Y

def rectify(x):
  return T.maximum(x, 0.)

def RMSprop(cost, params, dtype, lr=0.001, rho=0.9, epsilon=1e-6):
  grads = T.grad(cost=cost, wrt=params)
  updates = []
  for p, g in zip(params, grads):
    acc = theano.shared(p.get_value() * 0.)
    acc_new = rho * acc + (1 - rho) * g ** 2
    gradient_scaling = T.sqrt(acc_new + epsilon)
    g = g / gradient_scaling
    p_ = p - lr * g
    
    acc, acc_new, p, p_ = cast_4(acc, acc_new, p, p_, dtype)
    
    updates.append((acc, acc_new))
    updates.append((p, p_))
  return updates

def momentum(cost, params, dtype, learning_rate, momentum):
  grads = theano.grad(cost, params)
  updates = []
  
  for p, g in zip(params, grads):
    mparam_i = theano.shared(np.zeros(p.get_value().shape, dtype=dtype))
    v = momentum * mparam_i - learning_rate * g

    mparam_i = mparam_i.astype(dtype)
    v = v.astype(dtype)
    p = T.cast(p, dtype=dtype)
    
    updates.append((mparam_i, v))
    updates.append((p, p + v))

  return updates

def Adam(cost, params, dtype, lr=0.001, b1=0.1, b2=0.001, e=1e-8):
  updates = []
  grads = T.grad(cost, params)
  i = theano.shared(floatX(0., dtype))
  i_t = i + 1.
  i_t = T.cast(i_t, dtype)
  fix1 = 1. - (1. - b1)**i_t
  fix2 = 1. - (1. - b2)**i_t
  lr_t = lr * (T.sqrt(fix2) / fix1)
  for p, g in zip(params, grads):
      m = theano.shared(p.get_value() * 0.)
      v = theano.shared(p.get_value() * 0.)
      m_t = (b1 * g) + ((1. - b1) * m)
      v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
      g_t = m_t / (T.sqrt(v_t) + e * T.sqrt(fix2))
      p_t = p - (lr_t * g_t)
      m_t = T.cast(m_t, dtype)
      v_t = T.cast(v_t, dtype)
      p_t = T.cast(p_t, dtype)
      updates.append((m, m_t))
      updates.append((v, v_t))
      updates.append((p, p_t))
  updates.append((i, i_t))
  return updates

def model(x, w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o):
  # Two Convolutional Layers
  c1 = rectify(conv2d(x, w_c1) + b_c1.dimshuffle('x', 0, 'x', 'x'))
  p1 = pool_2d(c1, (3, 3), ignore_border = False)

  c2 = rectify(conv2d(p1, w_c2) + b_c2.dimshuffle('x', 0, 'x', 'x'))
  p2 = pool_2d(c2, (2, 2), ignore_border = False)
  
  # Dense Layer
  p2_flat = p2.flatten(2)
  h3 = rectify(T.dot(p2_flat, w_h3) + b_h3)

  # Output Layer
  p_y_given_x = T.nnet.softmax(T.dot(h3, w_o) + b_o)
  return p_y_given_x

def init_variables(x, t, params, dtype):
  p_y_given_x = model(x, *params)
  print(p_y_given_x.shape)
	
  # provides only the first row/col of data
  y = T.argmax(p_y_given_x, axis=1)

  cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, t))

  #updates = momentum(cost, params, dtype, learning_rate=0.05, momentum=0.9) # default learning rate: 0.01
  updates = RMSprop(cost, params, dtype)
  #updates = Adam(cost, params, dtype)

  train = theano.function([x, t], cost, updates=updates, allow_input_downcast=True)
  predict = theano.function([x], y, allow_input_downcast=True)

  return p_y_given_x, y, cost, updates, train, predict
