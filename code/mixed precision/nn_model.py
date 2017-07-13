import theano
import theano.tensor as T
import numpy as np

import load

from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d

def floatX(X, dtype):
    return np.asarray(X, dtype=dtype)

def init_weights(shape, dtype):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01, dtype))

def rectify(x):
    return T.maximum(x, 0.)

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

def momentum(cost, params, dtype, learning_rate, momentum):
    grads = theano.grad(cost, params)
    updates = []
    
    for p, g in zip(params, grads):
        mparam_i = theano.shared(np.zeros(p.get_value().shape, dtype=dtype))
        v = momentum * mparam_i - learning_rate * g

        mparam_i = mparam_i.astype(dtype)
        v = v.astype(dtype)
        p = T.cast(p, dtype=dtype)
        
        updates.append((mparam_i, v))
        updates.append((p, p + v))

    return updates

def model(x, w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o):
    c1 = rectify(conv2d(x, w_c1) + b_c1.dimshuffle('x', 0, 'x', 'x'))
    p1 = pool_2d(c1, (3, 3), ignore_border = False)

    c2 = rectify(conv2d(p1, w_c2) + b_c2.dimshuffle('x', 0, 'x', 'x'))
    p2 = pool_2d(c2, (2, 2), ignore_border = False)

    p2_flat = p2.flatten(2)
    h3 = rectify(T.dot(p2_flat, w_h3) + b_h3)

    p_y_given_x = T.nnet.softmax(T.dot(h3, w_o) + b_o)
    return p_y_given_x

def init_variables(x, t, params, dtype):
	p_y_given_x = model(x, *params)
	y = T.argmax(p_y_given_x, axis=1)

	cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, t))

	updates = momentum(cost, params, dtype, learning_rate=0.01, momentum=0.9)

	# compile theano functions
	train = theano.function([x, t], cost, updates=updates, allow_input_downcast=True)
	predict = theano.function([x], y, allow_input_downcast=True)

	return p_y_given_x, y, cost, updates, train, predict

def train_iteration(x_train, t_train, train, batch_size=50, num_iterations=50):
	'''
	@x_train: train data x
	@t_train: train data y
	@train: the function needs to be trained
	return the updated train function
	''' 
	for i in range(num_iterations):
	    print "iteration %d" % (i + 1)
	    for start in range(0, len(x_train), batch_size):
	    	start_time = time.time()
	        x_batch = x_train[start:start + batch_size]
	        t_batch = t_train[start:start + batch_size]
	        cost = train(x_batch, t_batch)
	        print("--- %s seconds ---" % (time.time() - start_time))
	return train

