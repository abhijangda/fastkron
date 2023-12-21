#include <iostream>
#include <cassert>
#include <functional>

#pragma once

class Matrix {
  uint32_t rows;
  uint32_t cols;
public:
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

  void* ptr() const {return data;}

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

  uint32_t hash() {
    return std::hash<uint>()(m()) ^ std::hash<uint>()(n());
  }
};

template<typename T, uint32_t MaxSize>
class StackArray {
public:
  T array[MaxSize];
  uint32_t n;

  StackArray() {
    for (uint32_t i = 0; i < MaxSize; i++) {
      array[i] = T();
    }
  }

public:
  StackArray(const T* ptrs, uint32_t n) : n(n) {
    if (ptrs) {
      for (uint32_t i = 0; i < n; i++) {
        array[i] = ptrs[i];
      }
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
    assert(len <= n);
    T ptrs[len];
    for (uint32_t i = 0; i < len; i++) {
      ptrs[i] = array[i + start];
    }

    return StackArray<T, MaxSize>(ptrs, len);
  }

  uint32_t len() const {return n;}

  StackArray(const StackArray& x) : StackArray (&x.array[0], x.len()) {}
};

class MatrixArray : public StackArray<Matrix, 64> {
  static const uint32_t MaxSize = 64;
  using Base = StackArray<Matrix, 64>;

  MatrixArray(StackArray<Matrix, MaxSize> arr) : Base(arr) {}

public:
  MatrixArray(uint32_t n, const uint32_t* ms, const uint32_t* ns, void* const* ptrs) : 
    Base(nullptr, n) {
    assert (n < MaxSize);
    for (uint32_t i = 0; i < n; i++) {
      Base::array[i] = Matrix(ms[i], ns[i], ptrs ? ptrs[i] : nullptr);
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