import time
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d
srng = RandomStreams()

def floatX(X, dtype0):
    return np.asarray(X, dtype=dtype0)

def init_weights(shape, dtype0):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01, dtype0))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(dtype, X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=dtype)
        X /= retain_prob
        X = T.cast(X, dtype=dtype)
    return X

def cast_4(trX, teX, trY, teY, dtype):
    trX = T.cast(trX, dtype=dtype)
    teX = T.cast(teX, dtype=dtype)
    trY = T.cast(trY, dtype=dtype)
    teY = T.cast(teY, dtype=dtype)
    return trX, teX, trY, teY

def RMSprop(cost, params, dtype, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        p_ = p - lr * g
        
        acc, acc_new, p, p_ = cast_4(acc, acc_new, p, p_, dtype)
        
        updates.append((acc, acc_new))
        updates.append((p, p_))
    return updates

def model(X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hidden, dtype):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = pool_2d(l1a, (2, 2), ignore_border = False)
    l1 = dropout(dtype,l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = pool_2d(l2a, (2, 2), ignore_border = False)
    l2 = dropout(dtype,l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = pool_2d(l3a, (2, 2), ignore_border = False)
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(dtype,l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(dtype,l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

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

    w = init_weights((32, 1, 3, 3),dtype0)
    w2 = init_weights((64, 32, 3, 3),dtype0)
    w3 = init_weights((128, 64, 3, 3),dtype0)
    w4 = init_weights((128 * 3 * 3, 625),dtype0)
    w_o = init_weights((625, 10),dtype0)

    trX, trY, X, Y = cast_4(trX, trY, X, Y, dtype0)
    
    noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, w_o, 0.2, 0.5, dtype0)
    l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, w_o, 0., 0., dtype0)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
    params = [w, w2, w3, w4, w_o]
    updates = RMSprop(cost, params, dtype0, lr=0.001)
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

    # train_model with mini-batch training
    for i in range(1):
        start_time = time.time()
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            print(start, ' ', end)
            cost = train(trX[start:end], trY[start:end])
            print("--- %s seconds ---" % (time.time() - start_time))
    pred_teX = predict(teX)
    print(np.mean(np.argmax(teY, axis=1) == pred_teX))
    
    w = theano.shared(value = np.asarray(w.eval(), dtype = dtype1), name = 'w', borrow = True)
    w2 = theano.shared(value = np.asarray(w2.eval(), dtype = dtype1), name = 'w2', borrow = True)
    w3 = theano.shared(value = np.asarray(w3.eval(), dtype = dtype1), name = 'w3', borrow = True)
    w4 = theano.shared(value = np.asarray(w4.eval(), dtype = dtype1), name = 'w4', borrow = True)
    w_o = theano.shared(value = np.asarray(w_o.eval(), dtype = dtype1), name = 'w_o', borrow = True)
    
    trX, trY, X, Y = cast_4(trX, trY, X, Y, dtype1)

    noise_l1 = T.cast(noise_l1, dtype=dtype1)
    noise_l2 = T.cast(noise_l2, dtype=dtype1)
    noise_l3 = T.cast(noise_l3, dtype=dtype1)
    noise_l4 = T.cast(noise_l4, dtype=dtype1)
    noise_py_x = T.cast(noise_py_x, dtype=dtype1)
    l1 = T.cast(l1, dtype=dtype1)
    l2 = T.cast(l2, dtype=dtype1)
    l3 = T.cast(l3, dtype=dtype1)
    l4 = T.cast(l4, dtype=dtype1)

    for i in range(1):
        start_time = time.time()
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            print(start, ' ', end)
            cost = train(trX[start:end], trY[start:end])
            print("--- %s seconds ---" % (time.time() - start_time))
    pred_teX = predict(teX)
    print(np.mean(np.argmax(teY, axis=1) == pred_teX))
    print("Finished!")

if __name__ == "__main__":
  main()
