from ctypes import c_int, c_float, Union
import numpy as np

class value_u(Union):
  _fields_ = [("as_int", c_int), ("as_float", c_float)]

def create_bitmask(bitsize):
  ''' create bitmask for bitwise operations
      @param bitsize: number of bits that will be 1's in bitmask
      @return bitmask
  '''
  bitmask = 1;
  for bit in range(bitsize):
    bitmask = bitmask << 1
    bitmask = bitmask + 1  
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

def main():
  x = 1.96875
  print truncate(x, 10)

if __name__ == "__main__":
  main()
