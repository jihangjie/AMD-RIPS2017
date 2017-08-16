from helper_files.load_svhn import load_svhn
from by_batch.cnn_train import iterate_train

def load_data():
  train_x, train_y, test_x, test_y = load_svhn()
  train_x = train_x.reshape(train_x.shape[0], 1, 32, 32)
  test_x = test_x.reshape(test_x.shape[0], 1, 32, 32)
  return train_x, train_y, test_x, test_y

def main():
  train_x, train_y, test_x, test_y = load_data()

  # change params as needed to test different conditions
  # these are all optional parameters, but defaults are shown here
  bitsize = 32
  savename = "svhn_FL{}".format(bitsize)
  perturbation = 0.
  batch_size = 128
  dense_units = 100
  iterate_train(train_x, test_x, train_y, test_y, bitsize, savename,
    preturbation, batch_size, dense_units)

if __name__ == "__main__":
  main()
