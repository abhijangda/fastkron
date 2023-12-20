#include <functional>

#include "sliced_mul_shape.h"

std::size_t std::hash<Matrix>::operator()(const Matrix& m) const {
  return hash<uint>()(m.M) ^ hash<uint>()(m.N);
}