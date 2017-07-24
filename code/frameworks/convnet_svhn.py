#!/usr/bin/python

from load_svhn import load_data
from network.cnn_train import iterate_train

def main():
  train_x, train_y, test_x, test_y = load_data()

  train_x = train_x.reshape(train_x.shape[0], 1, 32, 32)
  test_x = test_x.reshape(test_x.shape[0], 1, 32, 32)

  numbits = 32

  iterate_train(train_x, test_x, train_y, test_y, numbits, "svhn_FL{}".format(numbits))

if __name__ == "__main__":
  main()
