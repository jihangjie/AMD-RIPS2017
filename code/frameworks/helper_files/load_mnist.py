import theano
import numpy as np
import os

datasets_dir = 'data/mnist/'

def load_data():
  train_x, test_x, train_y, test_y = mnist(onehot=True)
  train_x = train_x.reshape(-1, 1, 28, 28) 
  test_x = test_x.reshape(-1, 1, 28, 28) 
  return train_x, test_x, train_y, test_y

def one_hot(x,n):
  if type(x) == list:
    x = np.array(x)
  x = x.flatten()
  o_h = np.zeros((len(x),n))
  o_h[np.arange(len(x)),x] = 1
  return o_h

def mnist(ntrain=60000,ntest=10000,onehot=True):
  data_dir = datasets_dir
  try:
    fd1 = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    fd2 = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    fd3 = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    fd4 = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
  except:
    print "MNIST dataset not downloaded! Please run download_mnist.sh first."
    exit(1)

  loaded = np.fromfile(file=fd1,dtype=np.uint8)
  trX = loaded[16:].reshape((60000,28*28)).astype(float)

  loaded = np.fromfile(file=fd2,dtype=np.uint8)
  trY = loaded[8:].reshape((60000))

  loaded = np.fromfile(file=fd3,dtype=np.uint8)
  teX = loaded[16:].reshape((10000,28*28)).astype(float)

  loaded = np.fromfile(file=fd4,dtype=np.uint8)
  teY = loaded[8:].reshape((10000))

  trX = trX/255.
  teX = teX/255.

  trX = trX[:ntrain]
  trY = trY[:ntrain]

  teX = teX[:ntest]
  teY = teY[:ntest]

  if onehot:
    trY = one_hot(trY, 10)
    teY = one_hot(teY, 10)
  else:
    trY = np.asarray(trY)
    teY = np.asarray(teY)

  return trX,teX,trY,teY
