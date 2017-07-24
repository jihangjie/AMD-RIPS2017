#!/usr/bin/python

from __future__ import division
import numpy as np
import scipy.io

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def convert_labels(labels):
  ''' convert labels numbers into boolean array
      @param labels: labels to be converted
      @return boolean array of labels
  '''
  bool_labels = np.zeros(shape = (len(labels), 10))
  for num, label in enumerate(labels):
    if label == 10:
      label = 0
    bool_labels[num][label] = 1
  return bool_labels

def convert_greyscale(img_data):
  ''' convert data from RGB images to greyscale
      @param img_data: image data
      @return reshaped, greyscale image data

  '''
  base, height, dim, num_images= img_data.shape
  grey_img = np.zeros(shape = (num_images, base*height))

  for num in range(num_images):
    img = rgb2gray(img_data[:, :, :, num])
    img = img.reshape(1, 1024)

    grey_img[num, :] = img

  return grey_img

def load_data():
  ''' loads image data
      @param filename
      @return image data
  '''
  dataset_dir = "data/svhn/"

  # load datasets
  train_data = scipy.io.loadmat("{}train_32x32.mat".format(dataset_dir))
  test_data = scipy.io.loadmat("{}test_32x32.mat".format(dataset_dir))

  train_data_x = convert_greyscale(train_data['X'])
  test_data_x = convert_greyscale(test_data['X'])

  train_data_y = convert_labels(train_data['y'])
  test_data_y = convert_labels(test_data['y'])

  return train_data_x, train_data_y, test_data_x, test_data_y

if __name__ == "__main__":
  load_data()
