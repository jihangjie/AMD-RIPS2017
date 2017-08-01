import os, time
import theano
import theano.tensor as T
import numpy as np
import truncate

from cnn_model import init_weights, init_variables, floatX

truncate_vectorize = np.vectorize(truncate.truncate, otypes=[np.float32])

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

def iterate_train(trX, teX, trY, teY, numPrecision=32, savename="untitled", perturbation = 0.002):

  dtype0 = 'float32'

  X = T.ftensor4()
  Y = T.fmatrix()  

  w_c1 = init_weights((4, 1, 4, 4), dtype0, perturbation)
  b_c1 = init_weights((4,), dtype0, perturbation)
  w_c2 = init_weights((8, 4, 3, 3), dtype0, perturbation)
  b_c2 = init_weights((8,), dtype0, perturbation)
  w_h3 = init_weights((8 * 4 * 4, 100), dtype0, perturbation)
  b_h3 = init_weights((100,), dtype0, perturbation)
  w_o = init_weights((100, 10), dtype0, perturbation)
  b_o = init_weights((10,), dtype0, perturbation)
  dw_c1 = theano.shared(floatX(np.zeros((4, 1, 4, 4)), dtype0))
  db_c1 = theano.shared(floatX(np.zeros((4,)), dtype0))
  dw_c2 = theano.shared(floatX(np.zeros((8, 4, 3, 3)), dtype0))
  db_c2 = theano.shared(floatX(np.zeros((8,)), dtype0))
  dw_h3 = theano.shared(floatX(np.zeros((8 * 4 * 4, 100)), dtype0))
  db_h3 = theano.shared(floatX(np.zeros((100,)), dtype0))
  dw_o = theano.shared(floatX(np.zeros((100, 10)), dtype0))
  db_o = theano.shared(floatX(np.zeros((10, )), dtype0))
  
  trX, trY, X, Y = cast_4(trX, trY, X, Y, dtype0)

  params = [w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o]
  oldGrad = [dw_c1, db_c1, dw_c2, db_c2, dw_h3, db_h3, dw_o, db_o]

  py_x, y_x, cost, updates, train, predict = init_variables(0, X, Y, params, dtype0, oldGrad)

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
    for i in range(50):
      t = time.time()
      for start in range(0, len(trX), batch_size):
        x_batch = trX[start:start + batch_size]
        t_batch = trY[start:start + batch_size]
        cost = train(x_batch, t_batch)
        # truncate all params
        t1 = time.time()
        w_c1 = truncate(w_c1, numPrecision)
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
        db_o = truncate(db_o, numPrecision)
      print("Iteration {}: {} seconds".format(i, time.time() - t))
      accuracy = np.mean(np.argmax(teY, axis=1) == predict(teX))
      if accuracy > 0.2 and accuracy - oldAccuracy < 0.02 and tempChanged == False:
        py_x, y_x, cost, updates, train, predict = init_variables(1, X, Y, params, dtype0, oldGrad)
        print("Momentum changing from 0.9 to 0.5")
      oldAccuracy = accuracy
      print(accuracy)
      f.write("{}\n".format(accuracy))
  
  print("Finished!")
