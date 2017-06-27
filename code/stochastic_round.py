# implements stochastic rounding

import tensorflow as tf
import numpy as np

import math
import random

def tensor_round(x, n_places):
  ''' rounds x to n_places number of decimal places
  @param x tensor
  @param n_places number of decimal places to round to
  @return rounded tensor
  '''
  x_matrix = x.eval()
  x_matrix = np.round(x_matrix, decimals=n_places)
  x.assign(x_matrix)
  return x

def prob_round(x):
  sign = np.sign(x)
  x = abs(x)
  is_up = random.random() < x - int(x)
  if is_up:
    round_func = math.ceil
  else:
    round_func = math.floor
  return sign * round_func(x)

def main():
  # for testing purposes only
  x = 6.5
  rounded = [prob_round(x) for i in range(100000)]
  print(sum(rounded)/100000)

if __name__ == "__main__":
  main()
