#This version has both dropout and momentum
import theano
import theano.tensor as T
import numpy as np
from theano import pp
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d
import math

def floatX(X, dtype):
  return np.asarray(X, dtype=dtype)

def init_weights(inputShape, shape, dtype, perturbation = 0.):
  np.random.seed(42)
  #return theano.shared(floatX(np.random.randn(*shape) * math.sqrt(2 / inputShape), dtype))
  return theano.shared(floatX(np.random.randn(*shape) * .01 + perturbation, dtype))
  #return theano.shared(floatX(np.random.randn(*shape) * .01, dtype))

def rectify(x):
  return T.maximum(x, 0.)

def momentum(cost, params, dtype, oldGrad, dropFilter, learning_rate = 0.05, momentum = 0.5):
  grads = theano.grad(cost, params) 
  updates = []
  
  for p, g, o, f in zip(params, grads, oldGrad, dropFilter):
    mparam_i = o
    v = momentum * mparam_i - np.multiply(learning_rate * g, f)

    mparam_i = mparam_i.astype(dtype)
    v = v.astype(dtype)
    p = T.cast(p, dtype=dtype)
    updates.append((o, v))
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

def model(x, ifTesting, hw_c1, hw_c2, hw_h3, hw_o, w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o):
  # Two Convolutional Layers
  if ifTesting == 0:
    c1 = rectify(conv2d(x, np.multiply(w_c1, hw_c1)) + b_c1.dimshuffle('x', 0, 'x', 'x'))
    p1 = pool_2d(c1, (2, 2), ignore_border = False)
    
    c2 = rectify(conv2d(p1, np.multiply(w_c2, hw_c2)) + b_c2.dimshuffle('x', 0, 'x', 'x'))
    p2 = pool_2d(c2, (2, 2), ignore_border = False)
    
    # Dense Layer
    p2_flat = p2.flatten(2)
    h3 = rectify(T.dot(p2_flat, np.multiply(w_h3, hw_h3)) + b_h3)
    
    # Output Layer
    p_y_given_x = T.nnet.softmax(T.dot(h3, w_o) + b_o)
  else:
    c1 = rectify(conv2d(x, np.multiply(w_c1, np.float32(0.9))) + b_c1.dimshuffle('x', 0, 'x', 'x'))
    p1 = pool_2d(c1, (2, 2), ignore_border = False)
    
    c2 = rectify(conv2d(p1, np.multiply(w_c2, np.float32(0.9))) + b_c2.dimshuffle('x', 0, 'x', 'x'))
    p2 = pool_2d(c2, (2, 2), ignore_border = False)
    
    # Dense Layer
    p2_flat = p2.flatten(2)
    h3 = rectify(T.dot(p2_flat, np.multiply(w_h3, np.float32(0.9))) + b_h3)
    
    # Output Layer
    p_y_given_x = T.nnet.softmax(T.dot(h3, np.multiply(w_o, np.float32(0.9))) + b_o)
  return p_y_given_x

def init_variables(previousTime, x, t, params, hw_c1, hw_c2, hw_h3, hw_o, dtype, oldGrad):
  p_y_given_x = model(x, 0, hw_c1, hw_c2, hw_h3, hw_o, *params)
  y = T.argmax(p_y_given_x, axis=1)

  cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, t))
  if previousTime == 0:
    #updates = Adam(cost, params, dtype, lr=0.005) # default learning rate: 0.01
    updates = momentum(cost, params, dtype, oldGrad, [hw_c1, 1, hw_c2, 1, hw_h3, 1, hw_o, 1], learning_rate=0.05, momentum=0.5)
  else:
    #updates = Adam(cost, params, dtype, lr=0.005)
    updates = momentum(cost, params, dtype, oldGrad, [hw_c1, 1, hw_c2, 1, hw_h3, 1, hw_o, 1], learning_rate=0.05, momentum=0.9)
  train = theano.function([x, t], cost, updates=updates, allow_input_downcast=True)
  
  p_y_given_xTest = model(x, 1, hw_c1, hw_c2, hw_h3, hw_o, *params)
  yTest = T.argmax(p_y_given_xTest, axis=1)
  predict = theano.function([x], yTest, allow_input_downcast=True)

  return p_y_given_x, y, cost, updates, train, predict
