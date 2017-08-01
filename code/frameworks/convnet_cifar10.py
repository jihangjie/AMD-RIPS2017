#!/usr/bin/python

import theano
import numpy as np
from load_cifar10 import cifar10
from network.cnn_train import iterate_train

def main():
  trX, trY, teX, teY = cifar10(dtype=theano.config.floatX)
  trX = trX.reshape(-1, 1, 32, 32)
  teX = teX.reshape(-1, 1, 32, 32)
  
  for bitsize in [12, 13, 14, 15, 16, 18, 20, 32]:
    for perturbation in [0.002]:
      iterate_train(trX, teX, trY, teY, bitsize, "cifar10_FL{}".format(bitsize), perturbation)

if __name__ == "__main__":
  main()
