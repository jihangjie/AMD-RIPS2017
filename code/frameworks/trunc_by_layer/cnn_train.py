import os, time
import theano
import theano.tensor as T
import numpy as np
import math
import truncate

from cnn_model import *

truncate_vectorize = np.vectorize(truncate.truncate, otypes=[np.float32])

def truncate(x, bitsize=32):
  value = x.eval()
  value = truncate_vectorize(value, bitsize)
  x.set_value(value)
  return x

# Use this version if x is a numpy array
def truncate_np(x, bitsize=32):
  value = truncate_vectorize(x, bitsize)
  return value

def cast_4(trX, trY, X, Y, dtype):
  trX = trX.astype(dtype)
  trY = trY.astype(dtype)
  X = T.cast(X, dtype=dtype)
  Y = T.cast(Y, dtype=dtype)
  return trX, trY, X, Y

def Adam_numeric(grads_value, params, dtype, bitsize = 32, lr=0.001, b1=0.1, b2=0.001, e=1e-8):
  i = theano.shared(floatX(0., dtype))
  i_t = i.get_value() + 1.
  fix1 = 1. - (1. - b1)**i_t
  fix2 = 1. - (1. - b2)**i_t
  lr_t = lr * (np.sqrt(fix2) / fix1)

  for p, g in zip(params, grads_value):
    m = theano.shared(p.get_value() * 0.)
    v = theano.shared(p.get_value() * 0.)
    m_t = (b1 * g) + ((1. - b1) * m.get_value())
    v_t = (b2 * g**2) + ((1. - b2) * v.get_value())
    g_t = m_t / (np.sqrt(v_t) + e * np.sqrt(fix2))
    p_t = p.get_value() - (lr_t * g_t)
    m_t = truncate_np(m_t, bitsize)
    v_t = truncate_np(v_t, bitsize)
    p_t = truncate_np(p_t, bitsize)
    m.set_value(m_t)
    v.set_value(v_t)
    p.set_value(p_t)
  i.set_value(i_t)

def iterate_train(trX, teX, trY, teY, dense_units = 100, numPrecision = 32, savename="untitled", perturbation = 0.002):

  dtype = 'float32'

  X = T.ftensor4()
  Y = T.fmatrix()  

  trX, trY, X, Y = cast_4(trX, trY, X, Y, dtype)
  w_c1 = init_weights((4, 1, 4, 4), dtype)  
  b_c1 = init_weights((4,), dtype)
  w_c2 = init_weights((8, 4, 3, 3), dtype)
  b_c2 = init_weights((8,), dtype)
  

  w_c1 = init_weights((4, 1, 4, 4), dtype)
  b_c1 = init_weights((4,), dtype)
  w_c2 = init_weights((8, 4, 3, 3), dtype)
  b_c2 = init_weights((8,), dtype)
  w_h3 = init_weights((8 * 4 * 4, dense_units), dtype)
  b_h3 = init_weights((dense_units,), dtype)
  w_o = init_weights((dense_units, 10), dtype)
  b_o = init_weights((10,), dtype)

  params = [w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o]
  params_first_layer = [w_c1, b_c1]
  params_second_layer = [w_c2, b_c2]
  params_dense_layer = [w_h3, b_h3]
  params_output_layer = [w_o, b_o]

  p1 = first_layer(X, *params_first_layer)
  p2 = second_layer(p1, *params_second_layer)
  h3 = dense_layer(p2, *params_dense_layer)
  p_y_given_x = output_layer(h3, *params_output_layer)

  y = T.argmax(p_y_given_x, axis=1)

  cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, Y))

  grads = T.grad(cost, params)

  first_layer_fun = theano.function([X], p1)
  second_layer_fun = theano.function([p1], p2)
  dense_layer_fun = theano.function([p2], h3)
  output_layer_fun = theano.function([h3], p_y_given_x)
  cost_fun = theano.function([p_y_given_x, Y], cost, allow_input_downcast=True)

  print "--- Past cost_fun ---"
  grads_fun = theano.function([X, Y, cost], grads)
  print "--- Past grads_fun ---"

  #train = theano.function([X, Y], cost, updates=updates, allow_input_downcast=True)
  predict = theano.function([X], y, allow_input_downcast=True)

  # determine location to save data
  savefile = "data/output/{}.txt".format(savename)
  try:
    os.stat("./data/output")
  except:
    os.mkdir("./data/output")

  # train_model with mini-batch training
  batch_size = 128
  lr = 0.001
  with open(savefile, 'w+') as f:
    for i in range(50):
      t = time.time()
      for start in range(0, len(trX), batch_size):
        x_batch = trX[start:start + batch_size]
        t_batch = trY[start:start + batch_size]
        
        p1_value = first_layer_fun(x_batch)
        p1_value = truncate_np(p1_value, numPrecision)
        p2_value = second_layer_fun(p1_value)
        p2_value = truncate_np(p2_value, numPrecision)
        h3_value = dense_layer_fun(p2_value)
        h3_value = truncate_np(h3_value, numPrecision)
        p_y_given_x_value = output_layer_fun(h3_value)
        p_y_given_x_value = truncate_np(p_y_given_x_value, numPrecision)

        cost_value = cost_fun(p_y_given_x_value, t_batch)
        cost_value = truncate_np(cost_value, numPrecision)

        grads_value = grads_fun(x_batch, t_batch, cost_value)
        for j in range(len(grads_value)):
          grads_value[j] = truncate_np(grads_value[j], numPrecision)

        # Use an Adam method, truncation of the new values included
        Adam_numeric(grads_value, params, dtype, bitsize = numPrecision)
      
      print("Iteration {}: {} seconds".format(i, time.time() - t))
      accuracy = np.mean(np.argmax(teY, axis=1) == predict(teX))
      print(accuracy)
      f.write("{}\n".format(accuracy))
      print(w_c1.get_value())
      

  
  print("Finished!")
