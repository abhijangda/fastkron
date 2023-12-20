#include <functional>

#include "sliced_mul_shape.h"

std::size_t std::hash<Matrix>::operator()(const Matrix& m) const {
  return hash<uint>()(m.m()) ^ hash<uint>()(m.n());
}