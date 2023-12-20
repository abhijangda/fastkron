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

  Matrix(uint32_t rows, uint32_t cols, uint32_t* data) : 
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


template<>
struct std::hash<Matrix> {
  std::size_t operator()(const Matrix& m) const;
};