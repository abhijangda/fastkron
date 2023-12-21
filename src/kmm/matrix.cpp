#include <functional>

#include "kmm/matrix.h"

std::size_t std::hash<Factor>::operator()(const Factor& m) const {
  return hash<uint>()(m.p()) ^ hash<uint>()(m.q());
}