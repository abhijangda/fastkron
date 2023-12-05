#include <functional>

#include "sliced_mul_shape.h"

std::size_t std::hash<SlicedMulShape>::operator()(const SlicedMulShape& shape) const {
  return hash<uint>()(shape.Q) ^ hash<uint>()(shape.P) ^ hash<uint>()(shape.K);
}