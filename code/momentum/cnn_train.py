#This version has dropout, momentum, adam, 2 conv and 1 dense, stochastic truncation
import os, time
import theano
import theano.tensor as T
import numpy as np
import truncate

from cnn_model import init_weights, init_variables, floatX
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

np.random.seed(1234)
truncate_vectorize = np.vectorize(truncate.truncate_stochastic, otypes=[np.float32])
rng = np.random.RandomState(1234)
srng = RandomStreams(rng.randint(999999))

def truncate(x, bitsize=32):
  value = x.eval()
  value = truncate_vectorize(value, bitsize)
  x.set_value(value)
  return x

def cast_4(trX, trY, X, Y, dtype):
  trX = trX.astype(dtype)
  trY = trY.astype(dtype)
  X = T.cast(X, dtype=dtype)
  Y = T.cast(Y, dtype=dtype)
  return trX, trY, X, Y

def dropConv(x, probability = 0.5, dtype = 'float32'):
  x_array = x.eval()
  for a in range(0, x_array.shape[0]):
    for b in range(0, x_array.shape[1]):
      for c in range(0, x_array.shape[2]):
        for d in range(0, x_array.shape[3]):
          if np.random.uniform(0,1)>probability:
            x_array[a,b,c,d] = 1
          else:
            x_array[a,b,c,d] = 0
  x.set_value(x_array)
  return x

def dropDense(x, probability = 0.5, dtype = 'float32'):
  x_array = x.eval()
  for a in range(0, x_array.shape[0]):
    for b in range(0, x_array.shape[1]):
      if np.random.uniform(0,1)>probability:
        x_array[a,b] = 1
      else:
        x_array[a,b] = 0
  x.set_value(x_array)
  return x

def iterate_train(trX, teX, trY, teY, numPrecision=32, savename="untitled", perturbation = 0.002):

  dtype0 = 'float32'

  X = T.ftensor4()
  Y = T.fmatrix()
  
  w_c1 = init_weights(4*1*3*3,(4, 1, 3, 3), dtype0, perturbation)
  b_c1 = theano.shared(floatX(np.zeros((4,)), dtype0))
  w_c2 = init_weights(4*11*11,(8, 4, 3, 3), dtype0, perturbation)
  b_c2 = theano.shared(floatX(np.zeros((8,)), dtype0))
  w_h3 = init_weights(8*4*4, (288, 100), dtype0, perturbation)
  b_h3 =theano.shared(floatX(np.zeros((100,)), dtype0))
  w_o = init_weights(100, (100, 10), dtype0, perturbation)
  b_o = theano.shared(floatX(np.zeros((10, )), dtype0))
  #to store old gradient
  dw_c1 = theano.shared(floatX(np.zeros((4, 1, 3, 3)), dtype0))
  db_c1 = theano.shared(floatX(np.zeros((4,)), dtype0))
  dw_c2 = theano.shared(floatX(np.zeros((8, 4, 3, 3)), dtype0))
  db_c2 = theano.shared(floatX(np.zeros((8,)), dtype0))
  dw_h3 = theano.shared(floatX(np.zeros((288, 100)), dtype0))
  db_h3 = theano.shared(floatX(np.zeros((100,)), dtype0))
  dw_o = theano.shared(floatX(np.zeros((100, 10)), dtype0))
  db_o = theano.shared(floatX(np.zeros((10, )), dtype0))
  #dropout filter
  hw_c1 = theano.shared(floatX(np.ones((4, 1, 3, 3)), dtype0))
  hw_c2 = theano.shared(floatX(np.ones((8, 4, 3, 3)), dtype0))
  hw_h3 = theano.shared(floatX(np.ones((288, 100)), dtype0))
  hw_o = theano.shared(floatX(np.ones((100, 10)), dtype0))
  
  trX, trY, X, Y = cast_4(trX, trY, X, Y, dtype0)

  params = [w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o]
  oldGrad = [dw_c1, db_c1, dw_c2, db_c2, dw_h3, db_h3, dw_o, db_o]
  dropFilter = [hw_c1, hw_c2, hw_h3, hw_o]

  py_x, y_x, cost, updates, train, predict = init_variables(0, X, Y, params, hw_c1, hw_c2, hw_h3, hw_o, dtype0, oldGrad)

  # determine location to save data
  savefile = "data/output/{}.txt".format(savename)
  try:
    os.stat("./data/output")
  except:
    os.mkdir("./data/output")

  # train_model with mini-batch training
  batch_size = 128
  oldAccuracy = 0
  tempChanged = False
  with open(savefile, 'w+') as f:
    for i in range(500):
      t = time.time()
      #generate new dropout filters
      hw_c1 = dropConv(hw_c1, 0.1)
      hw_c2 = dropConv(hw_c2, 0.1)
      hw_h3 = dropDense(hw_h3, 0.1)
      hw_o = dropDense(hw_o, 0.1)
      for start in range(0, len(trX), batch_size):
        x_batch = trX[start:start + batch_size]
        t_batch = trY[start:start + batch_size]
        cost = train(x_batch, t_batch)
        # truncate all params
        '''w_c1 = truncate(w_c1, numPrecision)
        b_c1 = truncate(b_c1, numPrecision)
        w_c2 = truncate(w_c2, numPrecision)
        b_c2 = truncate(b_c2, numPrecision)
        w_h3 = truncate(w_h3, numPrecision)
        b_h3 = truncate(b_h3, numPrecision)
        w_o = truncate(w_o, numPrecision)
        b_o = truncate(b_o, numPrecision)
        dw_c1 = truncate(dw_c1, numPrecision)
        db_c1 = truncate(db_c1, numPrecision)
        dw_c2 = truncate(dw_c2, numPrecision)
        db_c2 = truncate(db_c2, numPrecision)
        dw_h3 = truncate(dw_h3, numPrecision)
        db_h3 = truncate(db_h3, numPrecision)
        dw_o = truncate(dw_o, numPrecision)
        db_o = truncate(db_o, numPrecision)'''
      print("Iteration {}: {} seconds".format(i, time.time() - t))
      accuracy = np.mean(np.argmax(teY, axis=1) == predict(teX))
      if accuracy > 0.2 and accuracy - oldAccuracy < 0.02 and tempChanged == False:
        py_x, y_x, cost, updates, train, predict = init_variables(1, X, Y, params, hw_c1, hw_c2, hw_h3, hw_o, dtype0, oldGrad)
        tempChanged = True
        print("Momentum changing from 0.9 to 0.5")
      oldAccuracy = accuracy
      print(accuracy)
      f.write("{}\n".format(accuracy))
  
  print("Finished!")
