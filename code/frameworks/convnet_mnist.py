from load_mnist import mnist

from network.cnn_train import iterate_train

def main():
  trX, teX, trY, teY = mnist(onehot=True)

  trX = trX.reshape(-1, 1, 28, 28)
  teX = teX.reshape(-1, 1, 28, 28)
  numbits=32

  iterate_train(trX, teX, trY, teY,numbits, "mnist_FL{}".format(numbits))

if __name__ == "__main__":
  main()
