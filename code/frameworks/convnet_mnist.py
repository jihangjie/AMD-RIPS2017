#!/usr/bin/python

from load_mnist import mnist
from network.cnn_train import iterate_train
import numpy as np

def main():
  trX, teX, trY, teY = mnist(onehot=True)
  trX = trX.reshape(-1, 1, 28, 28)
  teX = teX.reshape(-1, 1, 28, 28)

  for bitsize in [12, 13, 14, 15, 16, 18, 20]:
    for dense_units in [20, 50, 80, 100, 120]:
      print "dense units: ", dense_units
      iterate_train(trX, teX, trY, teY, bitsize, "mnist_FL{}_UNIT{}".format(bitsize, dense_units), dense_units = dense_units)
  
if __name__ == "__main__":
  main()
