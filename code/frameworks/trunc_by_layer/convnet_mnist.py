from load_mnist import mnist

from cnn_train import iterate_train

def main():
  trX, teX, trY, teY = mnist(onehot=True)

  trX = trX.reshape(-1, 1, 28, 28)
  teX = teX.reshape(-1, 1, 28, 28)
  numbits=32

  
  iterate_train(trX, teX, trY, teY, numPrecision = numbits)

if __name__ == "__main__":
  main()
