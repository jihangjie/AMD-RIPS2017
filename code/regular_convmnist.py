import time
import theano
import truncate
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import random
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d
srng = RandomStreams()
import os

datasets_dir = '/home/cvajiac/Downloads/'

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def mnist(ntrain=60000,ntest=10000,onehot=True):
	data_dir = os.path.join(datasets_dir,'MNIST/')
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	return trX,teX,trY,teY

def floatX(X, dtype0):
    return np.asarray(X, dtype=dtype0)

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

def first_layer(X, w, dtype, p_drop_conv):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = pool_2d(l1a, (2, 2), ignore_border = False)
    l1 = dropout(dtype,l1, p_drop_conv)
    return l1

def second_layer(l1, w2, dtype, p_drop_conv):
    l2a = rectify(conv2d(l1, w2))
    l2 = pool_2d(l2a, (2, 2), ignore_border = False)
    l2 = dropout(dtype,l2, p_drop_conv)
    return l2

def third_layer(l2, w3, dtype, p_drop_conv):
    l3a = rectify(conv2d(l2, w3))
    l3b = pool_2d(l3a, (2, 2), ignore_border = False)
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(dtype,l3, p_drop_conv)
    return l3

def fourth_layer(l3, w4, dtype, p_drop_hidden):
    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(dtype,l4, p_drop_hidden)
    return l4

def model(X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hidden, dtype):
    l1 = first_layer(X, w, dtype, p_drop_conv)
    T.cast(l1, dtype) #TODO: change this to a truncating function
    l2 = second_layer(l1, w2, dtype, p_drop_conv)
    T.cast(l2, dtype)
    #truncate_2d(w, 14)
    
    l3 = third_layer(l2, w3, dtype, p_drop_conv)
    T.cast(l3, dtype)
    #truncate_2d(w, 14)

    l4 = fourth_layer(l3, w4, dtype, p_drop_hidden)
    T.cast(l4, dtype)
    #truncate_2d(w, 14)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

def cast_6(trX, teX, trY, teY, X, Y, dtype):
    trX = trX.astype(dtype)
    teX = teX.astype(dtype)
    trY = trY.astype(dtype)
    teY = teY.astype(dtype)
    X = T.cast(X, dtype=dtype)
    Y = T.cast(Y, dtype=dtype)
    return trX, teX, trY, teY, X, Y

def train_model(trX, teX, trY, teY, X, Y, w, w2, w3, w4, w_o, dtype):
    noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, w_o, 0.2, 0.5, dtype)
    l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, w_o, 0., 0., dtype)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
    params = [w, w2, w3, w4, w_o]
    updates = RMSprop(cost, params, dtype, lr=0.001)
    
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

    # train_model with mini-batch training
    for totalEpoch in range(500):
        inx = np.random.randint(len(trY), size = 100)
        cost = train(trX[inx], trY[inx])
        idx = np.random.randint(len(teY), size = 1000)
        if totalEpoch % 10 == 0:
          print(np.mean(np.argmax(teY[idx], axis=1) == predict(teX[idx])))
          
    return trX, teX, trY, teY, X, Y, w, w2, w3, w4, w_o


def main(bitsize=32):
    bitsize = 1
    print "FL{}".format(bitsize)
    trX, teX, trY, teY = mnist(onehot=True)

    trX = trX.reshape(-1, 1, 28, 28)
    teX = teX.reshape(-1, 1, 28, 28)

    dtype0 = 'float32'
    dtype1 = 'float64'

    X = T.ftensor4()
    Y = T.fmatrix()

    w = init_weights((32, 1, 3, 3), dtype0)
    w2 = init_weights((64, 32, 3, 3), dtype0)
    w3 = init_weights((128, 64, 3, 3), dtype0)
    w4 = init_weights((128 * 3 * 3, 625), dtype0)
    w_o = init_weights((625, 10), dtype0)

    # truncate w's!
    w = truncate_4d(w, bitsize)
    w2 = truncate_4d(w2, bitsize)
    w3 = truncate_4d(w3, bitsize)
    w4 = truncate_2d(w4, bitsize)
    w_o = truncate_2d(w_o, bitsize)
    

    trX, teX, trY, teY, X, Y = cast_6(trX, teX, trY, teY, X, Y, dtype0)
    
    trX, teX, trY, teY, X, Y, w, w2, w3, w4, w_o = train_model(trX, teX, trY, teY, X, Y, w, w2, w3, w4, w_o, dtype0)
    
    w = theano.shared(value = np.asarray(w.eval(), dtype = dtype1), name = 'w', borrow = True)
    w2 = theano.shared(value = np.asarray(w2.eval(), dtype = dtype1), name = 'w2', borrow = True)
    w3 = theano.shared(value = np.asarray(w3.eval(), dtype = dtype1), name = 'w3', borrow = True)
    w4 = theano.shared(value = np.asarray(w4.eval(), dtype = dtype1), name = 'w4', borrow = True)
    w_o = theano.shared(value = np.asarray(w_o.eval(), dtype = dtype1), name = 'w_o', borrow = True)
    

if __name__ == "__main__":
  main()
