from batch_mnist import fetch_data
from by_layer.cnn_train_layers import iterate_train

def main():
  train_x, test_x, train_y, test_y = fetch_data()
  for numLayers in [1, 2, 3, 4, 5]:
    for bitsize in [12, 14, 16, 18, 20, 24, 32]:
      print("Number of Layers: ", numLayers)
      savename = "MNIST_FL{}_Layers{}".format(bitsize, numLayers)
      iterate_train(train_x, test_x, train_y, test_y, bitsize, savename,
        numLayers = numLayers)

if __name__ == "__main__":
  main()
