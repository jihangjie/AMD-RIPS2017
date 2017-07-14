from ctypes import c_int, c_float, Union
import numpy as np
import random
import math

class value_u(Union):
  _fields_ = [("as_int", c_int), ("as_float", c_float)]

def prob_round(x):
  is_round_down = random.random() < .5
  if is_round_down:
    x = x - 1
  return x

MANTISSA_BITS = 23 # For float32


def create_bitmask(bitsize, round_down = False):
  ''' create bitmask for bitwise operations
      @param bitsize: number of bits that will be 1's in bitmask
      @return bitmask
  '''
  bitmask = 1;
  for bit in range(bitsize - 1):
    bitmask = bitmask << 1
    bitmask = bitmask + 1  
  if round_down:
    bitmask = bitmask - 1
  bitmask = bitmask << (32 - bitsize)
  return bitmask


def truncate(value, bitsize):
  ''' sets value of float using 'bitsize' bits to store
      uses c-style union to access bits as both float & int for bitwise ops
      @param value: the value to be truncated and stored
      @param bitsize: the amount of bits to store a number
      @return truncated float
  '''
  bitmask = create_bitmask(bitsize)
  truncated = value_u()
  truncated.as_float = value
  truncated.as_int = truncated.as_int & bitmask
  return truncated.as_float

def compute_probability(value, bitsize):
  mant, exp = math.frexp(value)
  exp -= 1 # because python uses 1.xxx to store the number, while the mantissa returned by frexp is between 0.5 and 1
  return (value % 2**(9-bitsize)) / (2**(9-bitsize+exp)) #9-bitsize+exp = MANTISSA_BITS - shift

def truncate_stochastic(value, bitsize):
  ''' a stochastic version of truncating, truncate the number based on how close it is above/below
  '''
  mant, exp = math.frexp(value)
  rd = random.random()
  print rd
  round_up = rd > compute_probability(value, bitsize)
  print round_up
  bitmask = create_bitmask(bitsize)
  truncated = value_u()
  truncated.as_float = value
  truncated.as_int = truncated.as_int & bitmask
  if round_up:
    truncated.as_float += 2**(8-bitsize+exp) #MANTISSA_BITS-shift+1
  return truncated.as_float

def main():
  x = 1.96875
  print "regular:    {}".format(truncate(x, 11))
  for i in range(1,10):
    y = truncate_stochastic(x, 11)
  #print "stochastic: {}".format(y)

  #m = create_bitmask(15)
  #print bin(m)
  #print bin(2**32 - m - 1)
  #print bin(2)
  
    print "stochastic: {}".format(y)
  

if __name__ == "__main__":
  main()
