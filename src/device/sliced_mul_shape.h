#include <iostream>

#pragma once

struct Matrix {
  uint M;
  uint N;
  void* data;

  Matrix() : M(0), N(0), data(nullptr) {}

  Matrix(uint M, uint N) : M(M), N(N), data(nullptr) {}

  Matrix(uint M, uint N, void* data) : M(M), N(N), data(data) {}

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

  Matrix(const Matrix& m) : M(m.M), N(m.N), data(m.data) {}
};


template<>
struct std::hash<Matrix> {
  std::size_t operator()(const Matrix& m) const;
};