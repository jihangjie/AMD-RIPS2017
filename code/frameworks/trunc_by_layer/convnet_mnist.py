from load_mnist import mnist

from cnn_train import iterate_train

def fetch_data():
  trX, teX, trY, teY = mnist(onehot=True)
  trX = trX.reshape(-1, 1, 28, 28)
  teX = teX.reshape(-1, 1, 28, 28)
  return trX, teX, trY, teY

def main():
  trX, teX, trY, teY = fetch_data()
  
  iterate_train(trX, teX, trY, teY)

if __name__ == "__main__":
  main()
