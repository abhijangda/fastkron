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
  
  CUDA_DEVICE_HOST
  Matrix(uint32_t rows, uint32_t cols) : 
    rows(rows), cols(cols), ptr(nullptr) {}

  CUDA_DEVICE_HOST
  Matrix(uint32_t rows, uint32_t cols, void* data) : 
    rows(rows), cols(cols), ptr(data) {}

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
  template<typename T>
  CUDA_DEVICE_HOST
  Matrix row(uint32_t row) const {
    return Matrix(1, n(), data<T>(row * n()));
  }
  template<typename T>
  CUDA_DEVICE_HOST
  T* data(uint32_t idx) const {
    return ((T*)ptr) + idx;
  }
  template<typename T>
  CUDA_DEVICE_HOST
  T* data(uint32_t row, uint32_t col) const {
    return data<T>((row * n() + col));
  }

  template<typename T>
  CUDA_DEVICE_HOST
  void set(uint32_t row, uint32_t col, T val) {
    *(data<T>(row, col)) = val;
  }
  template<typename T>
  CUDA_DEVICE_HOST
  T at(uint32_t row, uint32_t col) {
    return *(data<T>(row, col));
  }
  template<typename T>
  CUDA_DEVICE_HOST
  void add(uint32_t row, uint32_t col, T val) {
    *(data<T>(row, col)) += val;
  }

  // template<typename T>
  // CUDA_DEVICE_HOST
  // Slice slice(uint32_t row, uint32_t numrows, uint32_t col, uint32_t numcols) const {
  //   //TODO: fix CUDA asserts 
  //   //assert(0 <= row && row + numrows < m()); 
  //   //assert(0 <= col && col + numcols < n()); 
  //   return Slice(row, col, numrows, numcols, *this);
  // }

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

template<typename T>
class Slice {
  const Matrix parent;
  //TODO: Create Coord2D
  uint32_t startrow;
  uint32_t startcol;
  uint32_t rows;
  uint32_t cols;
  uint32_t P;
  uint32_t TileP;
  T* ptr;

public:
  CUDA_DEVICE_HOST
  Slice(uint32_t startrow, uint32_t startcol, uint32_t rows, uint32_t cols,
        uint32_t P, uint32_t TileP, Matrix parent) :
    startrow(startrow), startcol(startcol), rows(rows), cols(cols),
    P(P), TileP(TileP), parent(parent), ptr(parent.data<T>(startrow, startcol)) {}

  CUDA_DEVICE_HOST
  const T* data(uint32_t row, uint32_t col, uint32_t tileP) const {
    uint32_t idx = row * parent.n();
    if (TileP == P) {
      idx += col;
    } else {
      idx += (col/TileP)*P + tileP + col%TileP;
    }
    return &ptr[idx];
  }

  CUDA_DEVICE_HOST
  uint32_t m() const {return rows;}
};

class ShiftShared : public Matrix {
public:
  CUDA_DEVICE_HOST
  ShiftShared(uint32_t rows, uint32_t cols, void* ptr) :
    Matrix(rows, cols, ptr) {}

  template<typename T, uint32_t N>
  CUDA_DEVICE_HOST
  void store(uint32_t row, uint32_t col, uint32_t TileP, uint32_t CRegRows, T elems[N]) {
    #pragma unroll
    for (uint i = 0; i < N; i++) {
      //TODO: refactor based on paper
      uint shk = col + i;
      uint shTileK = (shk/TileP)/CRegRows; //TODO:
      uint finalShK = (shk/TileP)*TileP + (shTileK + shk%TileP)%TileP;
      set<T>(row, finalShK, elems[i]);
    }
  }
};

class Factor : public Matrix {
public:
  Factor() : Matrix() {}
  CUDA_DEVICE_HOST
  Factor(uint32_t rows, uint32_t cols) :
    Matrix(rows, cols) {}
  CUDA_DEVICE_HOST
  Factor(uint32_t rows, uint32_t cols, void* data) : 
    Matrix(rows, cols, data) {}

  CUDA_DEVICE_HOST
  uint32_t p() const {return Matrix::m();}
  CUDA_DEVICE_HOST
  uint32_t q() const {return Matrix::n();}

protected:
  CUDA_DEVICE_HOST
  uint32_t m() {return Matrix::m();}
  CUDA_DEVICE_HOST
  uint32_t n() {return Matrix::n();}
};

template<class Base, typename T>
class DirectShared : public Base {
public:
  //TODO: Coord2D
  uint32_t tilerow, tilecol;

  CUDA_DEVICE_HOST
  DirectShared(uint32_t TileP, uint32_t TileQ, void* ptr, 
               uint32_t tilerow, uint32_t tilecol) :
    Base(TileP, TileQ, ptr), tilerow(tilerow), tilecol(tilecol) {}

  CUDA_DEVICE_HOST
  void store(uint32_t eIdx, uint32_t num, const T* elems) {
    #pragma unroll
    for (uint ve = 0; ve < num; ve++) {
      uint idx = eIdx + ve;
      Base::template set<T>(idx/Base::n(), idx%Base::n(), elems[ve]);
    }
  }

  CUDA_DEVICE_HOST
  void store(uint32_t row, uint32_t col, uint32_t num, const T* elems) {
    #pragma unroll
    for (uint ve = 0; ve < num; ve++) {
      Base::template set<T>(row, col + ve, elems[ve]);
    }
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

//Make this Tensor3D
template<typename T, uint32_t TileM_, uint32_t SliceM_, uint32_t SliceN_>
class YRegisters : public Matrix {
public:
  //TODO: change names based on paper
  T regs[TileM_][SliceM_][SliceN_];
  CUDA_DEVICE_HOST
  uint32_t TileM()  {return TileM_;}
  CUDA_DEVICE_HOST
  uint32_t SliceM() {return SliceM_;}
  CUDA_DEVICE_HOST
  uint32_t SliceN() {return SliceN_;}

public:
  CUDA_DEVICE_HOST
  YRegisters() : Matrix(SliceM_, SliceN_) {zero();}
  
  CUDA_DEVICE_HOST
  void zero() {
    #pragma unroll
    for (uint r = 0; r < TileM_; r++) {
    #pragma unroll
    for (uint i = 0; i < SliceM_; i++) {
    #pragma unroll
    for (uint j = 0; j < SliceN_; j++) {
      regs[r][i][j] = (T)0;
    }}}
  }

  CUDA_DEVICE_HOST
  void add(uint32_t r, uint32_t i, uint32_t j, T val) {
    regs[r][i][j] += val;
  }

  CUDA_DEVICE_HOST
  T at(uint32_t r, uint32_t i, uint32_t j) {
    return regs[r][i][j];
  }
};

template<typename T, uint32_t TileM, uint32_t CRegRows_, uint32_t TileP_>
class XRegisters {
public:
  T regs[TileM][CRegRows_][TileP_];
  CUDA_DEVICE_HOST
  uint32_t TileP(){return TileP_;}
  CUDA_DEVICE_HOST
  uint32_t CRegRows() {return CRegRows_;}
public:
  CUDA_DEVICE_HOST
  XRegisters() {} 

  CUDA_DEVICE_HOST
  void set(uint32_t r, uint32_t i, uint32_t j, T val) {
    regs[r][i][j] = val;
  }

  CUDA_DEVICE_HOST
  T at(uint32_t r, uint32_t i, uint32_t j) {
    return regs[r][i][j];
  }
};

template<typename T, uint32_t TileP, uint32_t CRegCols>
class FRegisters {
public:
  T regs[TileP][CRegCols];

public:
  CUDA_DEVICE_HOST
  FRegisters() {}

  CUDA_DEVICE_HOST
  void set(uint32_t i, uint32_t j, T val) {
    regs[i][j] = val;
  }

  CUDA_DEVICE_HOST
  T at(uint32_t i, uint32_t j) {
    return regs[i][j];
  }
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