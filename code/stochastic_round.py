# implements stochastic rounding

import random
import math
import numpy as np

def prob_round(x):
  sign = np.sign(x)
  x = abs(x)
  is_up = random.random() < x-int(x)
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
