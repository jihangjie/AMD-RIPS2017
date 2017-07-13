import time
import theano
import theano.tensor as T
import numpy as np

from cnn_model import init_weights, init_variables

def cast_4(trX, trY, X, Y, dtype):
    trX = trX.astype(dtype)
    trY = trY.astype(dtype)
    X = T.cast(X, dtype=dtype)
    Y = T.cast(Y, dtype=dtype)
    return trX, trY, X, Y

def iterate_train(trX, teX, trY, teY):

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
        print("--- %s seconds ---" % (time.time() - t))
        pred_teX = predict(teX)
        print(np.mean(np.argmax(teY, axis=1) == pred_teX))
    
    print("Finished!")
