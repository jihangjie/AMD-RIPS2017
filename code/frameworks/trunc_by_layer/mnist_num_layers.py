from convnet_mnist import fetch_data
from cnn_train_layers import iterate_train

def main():
	trX, teX, trY, teY = fetch_data()
	for numLayers in [2, 3, 4, 5]:
		for bitsize in [12, 14, 16, 18, 20, 24, 32]:
			print("Number of Layers: ", numLayers)
			iterate_train(trX, teX, trY, teY, numPrecision = bitsize, savename = "MNIST_FL{}_Layers{}".format(bitsize, numLayers), numLayers = numLayers)
		

if __name__ == "__main__":
  main()