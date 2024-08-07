#include <iostream>
#include <cassert>
#include <functional>
#include <initializer_list> 

#include "utils/utils.h"
#include "fastkron.h"
#include "kmm/stackarray.h"
#include "config.h"

#pragma once

enum FastKronType {
  FastKronTypeNone,
  FastKronFloat,
  FastKronDouble,
  FastKronInt,
  FastKronHalf
};

static inline size_t sizeOfFastKronType(FastKronType t) {
  switch (t) {
    case FastKronTypeNone:
      return 0;
    case FastKronFloat:
      return 4;
    case FastKronDouble:
      return 8;
    case FastKronInt:
      return 4;
    case FastKronHalf:
      return 2;
  }
  return 0;
}

static inline std::string strOfFastKronType(FastKronType t) {
  switch(t) {
    case FastKronTypeNone:
      return "NONE";
    case FastKronFloat:
      return "f";
    case FastKronDouble:
      return "d";
    case FastKronInt:
      return "i";
    case FastKronHalf:
      return "h";
  }
  return "INVALID";
}

class Matrix {
protected:
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

  Matrix sameShape(void* ptr) const {
    return Matrix(rows, cols, ptr);
  }

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
  T* data(uint32_t row, uint32_t col, fastKronOp op) const {
    if (op == fastKronOp_N) return data<T>((row * n() + col));
    else if (op == fastKronOp_T) return data<T>((col * m() + row));
    return nullptr;
  }

  template<typename T>
  CUDA_DEVICE_HOST
  void set(uint32_t row, uint32_t col, fastKronOp op, T val) {
    *(data<T>(row, col, op)) = val;
  }
  template<typename T>
  CUDA_DEVICE_HOST
  T at(uint32_t row, uint32_t col, fastKronOp op) {
    return *(data<T>(row, col, op));
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

//TODO: Think about this
template<typename T, fastKronOp Op>
class Slice {
public:
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
    P(P), TileP(TileP), parent(parent), ptr(parent.data<T>(startrow, startcol, Op)) {}

  CUDA_DEVICE_HOST
  const T* data(uint32_t row, uint32_t col, uint32_t tileP) const {
    //TODO: use the other data method
    if (Op == fastKronOp_N) {
      uint32_t idx = row * parent.n();
      if (P <= TileP) {
        idx += col;
      } else {
        idx += (col/TileP)*P + tileP + col%TileP;
      }
      return &ptr[idx];
    } else if (Op == fastKronOp_T) {
      uint32_t idx = 0;
      if (P <= TileP) {
        idx = col;
      } else {
        idx += (col/TileP)*P + tileP + col%TileP;
      }

      idx = idx * parent.m() + row;
      return &ptr[idx];
    }
  }

  CUDA_DEVICE_HOST
  uint32_t data(uint32_t row, uint32_t slice, uint32_t elem, uint32_t tileP) const {
    //TODO: get common parts out
    if (Op == fastKronOp_N) {
      uint32_t idx = row * parent.n();
      idx += slice*P + tileP + elem;
      return idx;
    } else if (Op == fastKronOp_T) {
      uint32_t idx = slice*P + tileP + elem;
      idx = idx * parent.m() + row;
      return idx;
    }
  }

  CUDA_DEVICE_HOST
  const T* data(uint32_t idx) const {
    return &ptr[idx];
  }

  CUDA_DEVICE_HOST
  uint32_t m() const {return rows;}
  CUDA_DEVICE_HOST
  uint32_t numel() const {return rows * cols;}
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

  Factor sameShape(void* ptr) const {
    return Factor(rows, cols, ptr);
  }

protected:
  CUDA_DEVICE_HOST
  uint32_t m() {return Matrix::m();}
  CUDA_DEVICE_HOST
  uint32_t n() {return Matrix::n();}
};

template<uint32_t MaxSize>
class FactorArray : public StackArray<Factor, MaxSize> {
  using Base = StackArray<Factor, MaxSize>;
  FactorArray(StackArray<Factor, MaxSize> arr) : Base(arr) {}

public:
  FactorArray(uint32_t n, const uint32_t* ps, const uint32_t* qs, void* const* ptrs) : 
    Base(nullptr, n) {
    // assert (n < MaxSize);
    for (uint32_t i = 0; i < n; i++) {
      Base::array[i] = Factor(ps[i], qs[i], ptrs ? ptrs[i] : nullptr);
    }
  }

  FactorArray(const Factor* factors, uint32_t n) : Base(factors, n) {}
  FactorArray(std::initializer_list<Factor> facList) : Base(facList) {}

  CUDA_DEVICE_HOST
  Factor& operator[](int index) {
    return Base::array[index];
  }

  CUDA_DEVICE_HOST
  const Factor& operator[](int index) const {
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

struct MatrixComparator {
  bool operator()(const Matrix& a, const Matrix& b) const {
    if (a.m() < b.m()) return true;
    return a.n() < b.n(); 
  }
};