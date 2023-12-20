#include <iostream>

#pragma once

class Matrix {
  uint32_t rows;
  uint32_t cols;
  void* data;

public:
  Matrix() : rows(0), cols(0), data(nullptr) {}

  Matrix(uint32_t rows, uint32_t cols) : 
    rows(rows), cols(cols), data(nullptr)
  {}

  Matrix(uint32_t rows, uint32_t cols, void* data) : 
    rows(rows), cols(cols), data(data)
  {}

  uint32_t m() const {return rows;}
  uint32_t n() const {return cols;}

  uint32_t rowSize() const {return cols;}
  uint32_t colSize() const {return rows;}
  
  bool operator==(const Matrix& other) const {
    return m() == other.m() && n() == other.n();
  }

  bool operator!=(const Matrix& other) const {
    return !(*this == other);
  }
  
  friend std::ostream& operator<<(std::ostream &out, const Matrix& matrix) {
    out << matrix.m() << "x" << matrix.n();
    return out;
  }
};

template<typename T, uint32_t MaxSize>
class StackArray {
  T array[MaxSize];
  uint32_t n;

public:
  StackArray(T* ptrs, uint32_t n) : n(n) {
    for (uint32_t i = 0; i < n; i++) {
      array[i] = ptrs[i];
    }

    for (uint32_t i = n; i < MaxSize; i++) {
      array[i] = T();
    }
  }

  T& operator[](int index) {
    assert (index < n && index >= 0);
    return array[index];
  }

  T& operator[](uint32_t index) {
    assert (index < n);
    return array[index];
  }
};

template<>
struct std::hash<Matrix> {
  std::size_t operator()(const Matrix& m) const;
};