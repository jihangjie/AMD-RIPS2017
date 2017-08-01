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

def momentum(cost, params, dtype, oldGrad, learning_rate = 0.05, momentum = 0.5):
  grads = theano.grad(cost, params)
  updates = []
  
  for p, g, o in zip(params, grads, oldGrad):
    mparam_i = o
    v = momentum * mparam_i - learning_rate * g

    mparam_i = mparam_i.astype(dtype)
    v = v.astype(dtype)
    p = T.cast(p, dtype=dtype)
    
    updates.append((o, v))
    updates.append((p, p + v))

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

def init_variables(previousTime, x, t, params, dtype, oldGrad):
  p_y_given_x = model(x, *params)
	
  # provides only the first row/col of data
  y = T.argmax(p_y_given_x, axis=1)

  cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, t))
  if previousTime == 0:
    updates = momentum(cost, params, dtype, oldGrad, learning_rate=0.05, momentum=0.5) # default learning rate: 0.01
  else:
    updates = momentum(cost, params, dtype, oldGrad, learning_rate=0.05, momentum=0.9)
  train = theano.function([x, t], cost, updates=updates, allow_input_downcast=True)
  predict = theano.function([x], y, allow_input_downcast=True)

  return p_y_given_x, y, cost, updates, train, predict