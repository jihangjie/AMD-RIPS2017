import time
import theano
import theano.tensor as T
import numpy as np
import truncate

from cnn_model import init_weights, init_variables

def truncate_4d(x, bitsize=32):
  ''' truncate theano varibale
      @param x: theano var
      @return theano variable with truncated value
  '''
  value = x.eval()
  for dim1 in range(value.shape[0]):
    for dim2 in range(value.shape[1]):
      for dim3 in range(value.shape[2]):
        for dim4 in range(value.shape[3]):
          value[dim1][dim2][dim3][dim4] = truncate.truncate(value[dim1][dim2][dim3][dim4], bitsize)
  x.set_value(value)
  return x

def truncate_2d(x, bitsize=32):
  ''' truncate theano varibale
      @param x: theano var
      @return theano variable with truncated value
  '''
  value = x.eval()
  for row in range(value.shape[0]):
    for col in range(value.shape[1]):
      value[row][col] = truncate.truncate(value[row][col], bitsize)
  x.set_value(value)
  return x

def truncate_1d(x, bitsize=32):
  ''' truncate theano varibale
      @param x: theano var
      @return theano variable with truncated value
  '''
  value = x.eval()
  for num in range(value.shape[0]):
      value[num] = truncate.truncate(value[num], bitsize)
  return x



def cast_4(trX, trY, X, Y, dtype):
    trX = trX.astype(dtype)
    trY = trY.astype(dtype)
    X = T.cast(X, dtype=dtype)
    Y = T.cast(Y, dtype=dtype)
    return trX, trY, X, Y

def iterate_train(trX, teX, trY, teY,numPrecision):

    dtype0 = 'float32'
    dtype1 = 'float64'

    X = T.ftensor4()
    Y = T.fmatrix()

    w_c1 = init_weights((4, 1, 3, 3), dtype0)
    b_c1 = init_weights((4,), dtype0)
    w_c2 = init_weights((8, 4, 3, 3), dtype0)
    b_c2 = init_weights((8,), dtype0)
    w_h3 = init_weights((8 * 4 * 4, 100), dtype0)
    b_h3 = init_weights((100,), dtype0)
    w_o = init_weights((100, 10), dtype0)
    b_o = init_weights((10,), dtype0)
    
    trX, trY, X, Y = cast_4(trX, trY, X, Y, dtype0)

    #params = [w, w2, w3, w4, w_o]
    params = [w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o]

    py_x, y_x, cost, updates, train, predict = init_variables(X, Y, params, dtype0)

    # train_model with mini-batch training
    batch_size = 128
    for i in range(50):
        print "iteration %d" % (i + 1)
        t = time.time()
        for start in range(0, len(trX), batch_size):
            x_batch = trX[start:start + batch_size]
            t_batch = trY[start:start + batch_size]
            cost = train(x_batch, t_batch)
            w_c1 = truncate_4d(w_c1, numPrecision)
            b_c1 = truncate_1d(b_c1, numPrecision)
            w_c2 = truncate_4d(w_c2, numPrecision)
            b_c2 = truncate_1d(b_c2, numPrecision)
            w_h3 = truncate_2d(w_h3, numPrecision)
            b_h3 = truncate_1d(b_h3, numPrecision)
            w_o = truncate_2d(w_o, numPrecision)
            b_o = truncate_1d(b_o, numPrecision)
        print("--- %s seconds ---" % (time.time() - t))
        pred_teX = predict(teX)
        print(np.mean(np.argmax(teY, axis=1) == pred_teX))
    
    print("Finished!")
