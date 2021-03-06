import os, sys, time
import theano
import theano.tensor as T
import numpy as np

# allows files to be imported from parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import helper_files.truncate as truncate

from cnn_model import init_weights, init_biases, init_variables, floatX

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

def iterate_train(trX, teX, trY, teY, bitsize=32, savename="untitled", perturbation = 0., batch_size=128, dense_units = 100):

  X = T.ftensor4()
  Y = T.fmatrix()  

  dtype = 'float32'
  w_c1 = init_weights((4, 1, 4, 4), dtype)
  b_c1 = init_biases((4,), dtype)
  w_c2 = init_weights((8, 4, 3, 3), dtype)
  b_c2 = init_biases((8,), dtype)
  w_h3 = init_weights((8 * 4 * 4, dense_units), dtype)
  b_h3 = init_biases((dense_units,), dtype)
  w_o = init_weights((dense_units, 10), dtype)
  b_o = init_biases((10,), dtype)
  
  trX, trY, X, Y = cast_4(trX, trY, X, Y, dtype)

  params = [w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o]

  py_x, y_x, cost, updates, train, predict = init_variables(X, Y, params, dtype)

  # determine location to save data
  savefile = "data/output/{}_sqrt.txt".format(savename)
  try:
    os.stat("./data/output")
  except:
    os.mkdir("./data/output")

  # train_model with mini-batch training
  with open(savefile, 'w+') as f:
    for i in range(50):
      t = time.time()
      for start in range(0, len(trX), batch_size):
        x_batch = trX[start:start + batch_size]
        t_batch = trY[start:start + batch_size]
        cost = train(x_batch, t_batch)
        # truncate all params
        t1 = time.time()
        w_c1 = truncate(w_c1, bitsize)
        b_c1 = truncate(b_c1, bitsize)
        w_c2 = truncate(w_c2, bitsize)
        b_c2 = truncate(b_c2, bitsize)
        w_h3 = truncate(w_h3, bitsize)
        b_h3 = truncate(b_h3, bitsize)
        w_o = truncate(w_o, bitsize)
        b_o = truncate(b_o, bitsize)
      print("Iteration {}: {} seconds".format(i, time.time() - t))
      accuracy = np.mean(np.argmax(teY, axis=1) == predict(teX))
      print(accuracy)
      f.write("{}\n".format(accuracy))
  
  print("Finished!")
