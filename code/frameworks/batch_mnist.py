from helper_files.load_mnist import mnist
from by_batch.cnn_train import iterate_train

def fetch_data():
  train_x, test_x, train_y, test_y = mnist(onehot=True)
  train_x = train_x.reshape(-1, 1, 28, 28)
  test_x = test_x.reshape(-1, 1, 28, 28)
  return train_x, test_x, train_y, test_y

def main():
  train_x, test_x, train_y, test_y = fetch_data()

  # change params as needed to test different conditions
  # these are all optional parameters, but defaults are shown here
  bitsize = 32
  savename = "mnist_FL{}".format(bitsize)
  perturbation = 0.
  batch_size = 128
  dense_units = 100

  iterate_train(train_x, test_x, train_y, test_y, bitsize, savename, perturbation,
    batch_size, dense_units)

if __name__ == "__main__":
  main()
