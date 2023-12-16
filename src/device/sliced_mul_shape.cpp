#include <functional>

#include "sliced_mul_shape.h"

std::size_t  std::hash<Factor>::operator()(const Factor& f) const {
  return hash<uint>()(f.Q) ^ hash<uint>()(f.P);
}