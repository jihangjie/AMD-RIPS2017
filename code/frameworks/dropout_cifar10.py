import theano
import numpy as np

from helper_files.load_cifar10 import cifar10
from dropout.cnn_train import iterate_train

def load_data():
  train_x, train_y, test_x, test_y = cifar10(dtype=theano.config.floatX)
  train_x = train_x.reshape(-1, 1, 32, 32)
  test_x = test_x.reshape(-1, 1, 32, 32)
  return train_x, train_y, test_x, test_y

def main():
  train_x, train_y, test_x, test_y = load_data()
  
  # change params as needed to test different conditions
  # these are all optional parameters, but defaults are shown here
  bitsize = 32
  savename = "cifar10dropout_FL{}".format(bitsize)
  perturbation = 0.
  iterate_train(train_x, test_x, train_y, test_y, bitsize, savename, perturbation, dim=392)

if __name__ == "__main__":
  main()
