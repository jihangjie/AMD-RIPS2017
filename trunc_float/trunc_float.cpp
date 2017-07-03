/* trunc_float.cpp
 * implementation of truncated float class
 */

#include <cstdint>
#include <iostream>
#include "trunc_float.h"

trunc_float::trunc_float() {
  trunc_float(0, 32);
}

trunc_float::trunc_float(float value) {
  trunc_float(value, 32);
}

trunc_float::trunc_float(float value, int bitsize) {
  set_bitsize(bitsize);
  set_value(value);
}

trunc_float::~trunc_float() {}

int trunc_float::get_bitsize() {
  return bitsize;
}

float trunc_float::get_value() {
  return value;
}

void trunc_float::set_bitsize(int bitsize) {
  this->bitsize = bitsize;
  set_value(this->value);
}

void trunc_float::set_value(float value) {
/* sets value of float, uses this->bitsize bits to store
 * @param value: value to be truncated and stored
 */
  // create filter
  uint32_t filter = 0;
  for (int bit = 0; bit < this->bitsize - 1; bit++) {
    filter++;
    filter = filter << 1;
  }
  filter++;
  filter = filter << (32 - bitsize);

  // bitwise AND filter with value
  value_u truncated;
  truncated.as_float = value;
  truncated.as_int = truncated.as_int & filter;
  this->value = truncated.as_float;
}
