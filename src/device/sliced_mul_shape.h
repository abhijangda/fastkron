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
protected:
  T array[MaxSize];
  uint32_t n;

  StackArray() {
    for (uint32_t i = 0; i < MaxSize; i++) {
      array[i] = T();
    }
  }

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

  StackArray<T, MaxSize> sub(uint32_t start, uint32_t len) const {
    assert(len < n);
    T ptrs[len];
    for (uint32_t i = 0; i < len; i++) {
      ptrs[i] = array[i + start];
    }

    return StackArray<T, MaxSize>(ptrs, len);
  }
};

class MatrixArray : public StackArray<Matrix, 64> {
  static const uint32_t MaxSize = 64;
  using Base = StackArray<Matrix, 64>;

  MatrixArray(StackArray<Matrix, MaxSize> arr) {
    for (uint32_t i = 0; i < n; i++) {
      Base::array[i] = arr[i];
    }
  }

public:
  MatrixArray(uint32_t n, const uint32_t* ms, const uint32_t* ns, void* const* ptrs) {
    assert (n < MaxSize);
    for (uint32_t i = 0; i < n; i++) {
      Base::array[i] = Matrix(ms[i], ns[i], ptrs[i]);
    }
  }

  MatrixArray(Matrix* matrices, uint32_t n) : Base(matrices, n) {}

  Matrix& operator[](int index) {
    assert (index < Base::n && index >= 0);
    return Base::array[index];
  }

  Matrix operator[] (int index) const {
    assert (index < Base::n && index >= 0);
    return Base::array[index];
  }

  MatrixArray sub(uint32_t start, uint32_t len) const {
    return MatrixArray(Base::sub(start, len));
  }

  // Matrix& operator[](uint32_t index) {
  //   assert (index < Base::n);
  //   return Base::array[index];
  // }
};

template<>
struct std::hash<Matrix> {
  std::size_t operator()(const Matrix& m) const;
};