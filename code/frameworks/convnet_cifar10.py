#!/usr/bin/python

import theano
import numpy as np

from load_cifar10 import cifar10
from network.cnn_train import iterate_train

def main():
  # load data
  x_train, t_train, x_test, t_test = cifar10(dtype=theano.config.floatX)

  # reshape data
  x_train = x_train.reshape((x_train.shape[0], 1, 32, 32))
  x_test = x_test.reshape((x_test.shape[0], 1, 32, 32))
  print "x_train shape: {}".format(x_train.shape)
  print "x_test shape: {}".format(x_test.shape)
  print t_train[0]

  for bitsize in [12, 13, 14, 15, 16, 18, 20, 32]:
    iterate_train(x_train, x_test, t_train, t_test, bitsize, "cifar10_FL{}".format(bitsize))

if __name__ == "__main__":
  main()
