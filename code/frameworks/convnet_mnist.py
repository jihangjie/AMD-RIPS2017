from load_mnist import mnist

from network.cnn_train import iterate_train

import numpy as np

def main():
  trX, teX, trY, teY = mnist(onehot=True)

  trX = trX.reshape(-1, 1, 28, 28)
  teX = teX.reshape(-1, 1, 28, 28)
  perturbation=0.001

  print trY[4]

  for bitsize in [16, 18]:
    for perturbation in [0.002]:
      iterate_train(trX, teX, trY, teY, 32, "mnist_FL{}_PERT{}".format(bitsize, perturbation), perturbation)
  #for bitsize in [12, 13, 14, 15, 16, 18, 20]:
  #  for perturbation in np.arange(0, 0.01, 0.002):
  #    iterate_train(trX, teX, trY, teY, 32, "mnist_FL{}_PERT{}".format(bitsize, perturbation), perturbation)

  for bitsize in [32]:
    for perturbation in np.arange(0.006, 0.01, 0.002):
      iterate_train(trX, teX, trY, teY, 32, "mnist_FL{}_PERT{}".format(bitsize, perturbation), perturbation)
      
  
if __name__ == "__main__":
  main()
