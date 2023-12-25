#include <iostream>
#include <cassert>
#include <functional>

#include "config.h"

#pragma once

class Matrix {
  uint32_t rows;
  uint32_t cols;
public:
  void* ptr;

public:
  Matrix() : rows(0), cols(0), ptr(nullptr) {}

  Matrix(uint32_t rows, uint32_t cols) : 
    rows(rows), cols(cols), ptr(nullptr)
  {}

  Matrix(uint32_t rows, uint32_t cols, void* data) : 
    rows(rows), cols(cols), ptr(data)
  {}

  CUDA_DEVICE_HOST
  uint32_t m() const {return rows;}
  CUDA_DEVICE_HOST
  uint32_t n() const {return cols;}
  CUDA_DEVICE_HOST
  uint32_t numel() const {return rows * cols;}

  uint32_t rowSize() const {return cols;}
  uint32_t colSize() const {return rows;}
  
  CUDA_DEVICE_HOST
  void* data() const {return ptr;}

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

  uint32_t hash() const {
    return std::hash<uint>()(m()) ^ std::hash<uint>()(n());
  }
};

class Factor : public Matrix {
public:
  Factor() : Matrix() {}
  Factor(uint32_t rows, uint32_t cols) : Matrix(rows, cols) {}
  Factor(uint32_t rows, uint32_t cols, void* data) : Matrix(rows, cols, data) {}

  CUDA_DEVICE_HOST
  uint32_t p() const {return Matrix::m();}
  CUDA_DEVICE_HOST
  uint32_t q() const {return Matrix::n();}

  uint32_t m() = delete;
  uint32_t n() = delete;
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

  CUDA_DEVICE_HOST
  T& operator[](int index) {
    assert (index < n && index >= 0);
    return array[index];
  }

  CUDA_DEVICE_HOST
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

  CUDA_DEVICE_HOST
  uint32_t len() const {return n;}

  StackArray(const StackArray& x) : StackArray (&x.array[0], x.len()) {}
};

template<uint32_t MaxSize>
class FactorArray : public StackArray<Factor, MaxSize> {
  using Base = StackArray<Factor, MaxSize>;
  FactorArray(StackArray<Factor, MaxSize> arr) : Base(arr) {}

public:
  FactorArray(uint32_t n, const uint32_t* ps, const uint32_t* qs, void* const* ptrs) : 
    Base(nullptr, n) {
    assert (n < MaxSize);
    for (uint32_t i = 0; i < n; i++) {
      Base::array[i] = Factor(ps[i], qs[i], ptrs ? ptrs[i] : nullptr);
    }
  }

  FactorArray(const Factor* factors, uint32_t n) : Base(factors, n) {}

  CUDA_DEVICE_HOST
  Factor& operator[](int index) {
  #if defined(__NVCC__) || defined(__CUDACC__)
  #else
    assert (index < Base::n && index >= 0);
  #endif
    return Base::array[index];
  }

  CUDA_DEVICE_HOST
  const Factor& operator[](int index) const {
  #if defined(__NVCC__) || defined(__CUDACC__)
  #else
    assert (index < Base::n && index >= 0);
  #endif
    return Base::array[index];
  }

  FactorArray sub(uint32_t start, uint32_t len) const {
    return FactorArray(Base::sub(start, len));
  }
};

template<>
struct std::hash<Factor> {
  std::size_t operator()(const Factor& m) const;
};