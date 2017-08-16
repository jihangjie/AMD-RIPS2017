from helper_files.load_mnist import mnist
from dropout.cnn_train import iterate_train
import numpy as np

def main():
  train_x, test_x, train_y, test_y = mnist(onehot=True)
  train_x = train_x.reshape(-1, 1, 28, 28)
  test_x = test_x.reshape(-1, 1, 28, 28)

  # change params as needed to test different conditions
  # these are all optional parameters, but defaults are shown here
  bitsize = 32
  savename = "mnistdropout_FL{}".format(bitsize)
  perturbation = 0.
  iterate_train(train_x, test_x, train_y, test_y, 32, savename, perturbation, dim=288)

if __name__ == "__main__":
  main()
