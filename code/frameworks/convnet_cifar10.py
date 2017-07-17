import theano
import numpy as np

# user-defined
import load_cifar10
from network.cnn_train import iterate_train

def main():
  # load data
  x_train, t_train, x_test, t_test = load_cifar10.cifar10(dtype=theano.config.floatX)
  labels_test = np.argmax(t_test, axis=1)

  # reshape data
  x_train = x_train.reshape((x_train.shape[0], 1, 32, 32))
  x_test = x_test.reshape((x_test.shape[0], 1, 32, 32))

  bitsize = 32
  iterate_train(x_train, x_test, t_train, t_test, bitsize, "cifar10_FL{}".format(bitsize))

if __name__ == "__main__":
  main()
