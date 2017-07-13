import time
import theano
import theano.tensor as T
import numpy as np

from load import mnist

from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d

import nn_model

def cast_4(trX, trY, X, Y, dtype):
    trX = trX.astype(dtype)
    trY = trY.astype(dtype)
    X = T.cast(X, dtype=dtype)
    Y = T.cast(Y, dtype=dtype)
    return trX, trY, X, Y

def main():
    trX, teX, trY, teY = mnist(onehot=True)

    trX = trX.reshape(-1, 1, 28, 28)
    teX = teX.reshape(-1, 1, 28, 28)

    dtype0 = 'float32'
    dtype1 = 'float64'

    X = T.ftensor4()
    Y = T.fmatrix()

    w_c1 = nn_model.init_weights((4, 1, 3, 3), dtype0)
    b_c1 = nn_model.init_weights((4,), dtype0)
    w_c2 = nn_model.init_weights((8, 4, 3, 3), dtype0)
    b_c2 = nn_model.init_weights((8,), dtype0)
    w_h3 = nn_model.init_weights((8 * 4 * 4, 100), dtype0)
    b_h3 = nn_model.init_weights((100,), dtype0)
    w_o = nn_model.init_weights((100, 10), dtype0)
    b_o = nn_model.init_weights((10,), dtype0)
    
    trX, trY, X, Y = cast_4(trX, trY, X, Y, dtype0)

    #params = [w, w2, w3, w4, w_o]
    params = [w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o]

    py_x, y_x, cost, updates, train, predict = nn_model.init_variables(X, Y, params, dtype0)

    # train_model with mini-batch training
    batch_size = 128
    for i in range(30):
        print "iteration %d" % (i + 1)
        for start in range(0, len(trX), batch_size):
            x_batch = trX[start:start + batch_size]
            t_batch = trY[start:start + batch_size]
            cost = train(x_batch, t_batch)

    pred_teX = predict(teX)
    print(np.mean(np.argmax(teY, axis=1) == pred_teX))
    
    print("Finished!")

if __name__ == "__main__":
  main()
