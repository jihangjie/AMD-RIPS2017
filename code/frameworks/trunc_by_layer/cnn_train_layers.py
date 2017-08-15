import os, time
import theano
import theano.tensor as T
import numpy as np
import math
import truncate

from cnn_model_layers import *

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

def iterate_train(trX, teX, trY, teY, dense_units = 100, numPrecision = 32, savename="untitled", perturbation = 0.002, numLayers = 2):

  dtype = 'float32'

  X = T.ftensor4()
  Y = T.fmatrix()  

  trX, trY, X, Y = cast_4(trX, trY, X, Y, dtype)

  w_c1 = init_weights((784, dense_units), dtype)  
  b_c1 = init_weights((dense_units,), dtype)

  w_c_lst = []
  b_c_lst = []
  for i in range(numLayers - 1):
    w_c_lst.append(init_weights((dense_units, dense_units), dtype))
    b_c_lst.append(init_weights((dense_units,), dtype))

  w_o = init_weights((dense_units, 10), dtype)
  b_o = init_weights((10,), dtype)

  params = [w_c1, b_c1]
  for i in range(len(w_c_lst)):
    params.append(w_c_lst[i])
    params.append(b_c_lst[i])
  params.append(w_o)
  params.append(b_o)
  
  params_first_layer = [w_c1, b_c1]
  params_output_layer = [w_o, b_o]
  p1 = dense_layer(X, *params_first_layer)
  output_lst = [p1]
  for i in range(numLayers - 1):
    p = dense_layer(output_lst[-1], w_c_lst[i], b_c_lst[i])
    output_lst.append(p)

  p_y_given_x = output_layer(output_lst[-1], *params_output_layer)

  y = T.argmax(p_y_given_x, axis=1)

  cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, Y))

  grads = T.grad(cost, params)

  first_layer_fun = theano.function([X], p1)
  dense_layer_fun_lst = []
  for i in range(numLayers - 1):
    dense_layer_fun_lst.append(theano.function([output_lst[i]], output_lst[i+1]))
  output_layer_fun = theano.function([output_lst[-1]], p_y_given_x)
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
        for j in range(numLayers - 1):
          dense_layer_fun = dense_layer_fun_lst[j]
          output_value = dense_layer_fun(p1_value)
          output_value = truncate_np(output_value, numPrecision)
          p1_value = output_value
        p_y_given_x_value = output_layer_fun(output_value)
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
