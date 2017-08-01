#!/usr/bin/python

from load_mnist import mnist
from network.cnn_train import iterate_train
import numpy as np

def main():
  trX, teX, trY, teY = mnist(onehot=True)
  trX = trX.reshape(-1, 1, 28, 28)
  teX = teX.reshape(-1, 1, 28, 28)

  for bitsize in [12, 13, 14, 15, 16, 18, 20, 32]:
    for perturbation in [0.002]:
      iterate_train(trX, teX, trY, teY, 32, "mnist_FL{}".format(bitsize), perturbation)
  
if __name__ == "__main__":
  main()
