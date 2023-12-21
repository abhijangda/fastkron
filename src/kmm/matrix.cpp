#include <functional>

#include "kmm/matrix.h"

std::size_t std::hash<Matrix>::operator()(const Matrix& m) const {
  return hash<uint>()(m.m()) ^ hash<uint>()(m.n());
}