#include <iostream>

#pragma once

struct Factor {
  uint Q;
  uint P;

  Factor() {}

  Factor(uint Q, uint P) : Q(Q), P(P) {}

  bool operator==(const Factor& other) const {
    return Q == other.Q && P == other.P;
  }

  bool operator!=(const Factor& other) const {
    return !(*this == other);
  }
  
  friend std::ostream& operator<<(std::ostream &out, const Factor& factor) {
    out << factor.P << "x" << factor.Q;
    return out;
  }
};

struct Matrix {
  uint M;
  uint N;

  Matrix() {}

  Matrix(uint M, uint N) : M(M), N(N) {}

  bool operator==(const Matrix& other) const {
    return M == other.M && N == other.N;
  }

  bool operator!=(const Matrix& other) const {
    return !(*this == other);
  }
  
  friend std::ostream& operator<<(std::ostream &out, const Matrix& matrix) {
    out << matrix.M << "x" << matrix.N;
    return out;
  }
};


template<>
struct std::hash<Factor> {
  std::size_t operator()(const Factor& f) const;
};