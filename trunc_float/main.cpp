/* main.cpp
 * test driver for trunc_float class
 */

#include <iostream>
#include "trunc_float.h"

int main() {
  trunc_float F(1.6542, 18);
  std::cout << "Value is " << F.get_value() << ", Bitsize is " << F.get_bitsize() << std::endl;
  F.set_bitsize(12);
  std::cout << "Value is " << F.get_value() << ", Bitsize is " << F.get_bitsize() << std::endl;
  F.set_bitsize(18);
  std::cout << "Value is " << F.get_value() << ", Bitsize is " << F.get_bitsize() << std::endl;
  return 0;
}
