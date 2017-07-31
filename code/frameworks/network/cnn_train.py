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

def iterate_train(trX, teX, trY, teY, numPrecision=32, savename="untitled", perturbation = 0.):

  dtype0 = 'float32'
  dtype1 = 'float64'

  X = T.ftensor4()
  Y = T.fmatrix()  

  w_c1 = theano.shared(floatX(np.zeros((4, 1, 4, 4)) + perturbation, dtype0))
  b_c1 = theano.shared(floatX(np.zeros((4,)) + perturbation, dtype0))
  w_c2 = theano.shared(floatX(np.zeros((8, 4, 3, 3)) + perturbation, dtype0))
  b_c2 = theano.shared(floatX(np.zeros((8,)) + perturbation, dtype0))
  w_h3 = theano.shared(floatX(np.zeros((8 * 4 * 4, 100)) + perturbation, dtype0))
  b_h3 = theano.shared(floatX(np.zeros((100,)) + perturbation, dtype0))
  w_o = theano.shared(floatX(np.zeros((100, 10)) + perturbation, dtype0))
  b_o = theano.shared(floatX(np.zeros((10, )) + perturbation, dtype0))
  
  trX, trY, X, Y = cast_4(trX, trY, X, Y, dtype0)

  params = [w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o]

  py_x, y_x, cost, updates, train, predict = init_variables(X, Y, params, dtype0)

  # determine location to save data
  savefile = "data/output/{}.txt".format(savename)
  try:
    os.stat("./data/output")
  except:
    os.mkdir("./data/output")

  # train_model with mini-batch training
  batch_size = 128
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
      print("Iteration {}: {} seconds".format(i, time.time() - t))
      accuracy = np.mean(np.argmax(teY, axis=1) == predict(teX))
      print(accuracy)
      f.write("{}\n".format(accuracy))
  
  print("Finished!")
