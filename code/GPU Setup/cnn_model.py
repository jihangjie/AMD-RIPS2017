import theano
import theano.tensor as T
import numpy as np
import lasagne
import cPickle
import gzip
import sys
import os
import time

from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d

def floatX(X, dtype):
    return np.asarray(X, dtype=dtype)

def init_weights(shape, dtype):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01, dtype))

def model(x):
#def model(x, w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o) was not used since in general they cause dim errors
    l=lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=x)

    l=lasagne.layers.Conv2DLayer(l, num_filters=32,filter_size=(5,5),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    l=lasagne.layers.MaxPool2DLayer(l,pool_size=(2,2))
    #(3,3),ignore_border=False)

    l=lasagne.layers.Conv2DLayer(l, num_filters=32,filter_size=(5,5),nonlinearity=lasagne.nonlinearities.rectify)
    l=lasagne.layers.MaxPool2DLayer(l,pool_size=(2,2))

    #l=lasagne.layers.FlattenLayer(l,2)

    l=lasagne.layers.DenseLayer(lasagne.layers.dropout(l, p=.5), num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
    l=lasagne.layers.DenseLayer(lasagne.layers.dropout(l, p=.5), num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
    return l

def init_variables(x,t):
    l=model(x)
    p_y_given_x= lasagne.layers.get_output(l)
    test_prediction=lasagne.layers.get_output(l) #deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            t)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),t),
                                        dtype=theano.config.floatX)


    cost=lasagne.objectives.categorical_crossentropy(p_y_given_x, t)
    cost=cost.mean()
    params=lasagne.layers.get_all_params(l, trainable=True)
    updates=lasagne.updates.momentum(cost, params, learning_rate=0.01, momentum=0.9)
    
    #most errors are caused by these two lines
    train = theano.function([x,t], cost, updates=updates,allow_input_downcast=True)
    predict = theano.function([x,t],[test_loss,test_acc],allow_input_downcast=True)
    return train, predict
     
