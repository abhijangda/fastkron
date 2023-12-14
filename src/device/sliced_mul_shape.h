#include <iostream>

#pragma once

//TODO: Add NumFusedKernels also as a parameter to SlicedMulShape for compiledKernels map
//TODO: Add array of NumFusedKernels of KronCols and KronRows
struct SlicedMulShape {
  uint Q;
  uint P;
  uint K;
  uint M;
  uint NumFusedKerns;
  bool DistributeToGPUs;

  bool operator==(const SlicedMulShape& other) const {
    return P == other.P && Q == other.Q && K == other.K &&
    NumFusedKerns == other.NumFusedKerns &&
    DistributeToGPUs == other.DistributeToGPUs;
  }

  bool sameKronSize(const SlicedMulShape& other) const {
    return P == other.P && Q == other.Q;
  }

  bool isTileOf(const SlicedMulShape& other) const {
    return other.P == P && other.Q % Q == 0 && other.K % K == 0 && 
           other.NumFusedKerns == NumFusedKerns && 
           other.DistributeToGPUs == DistributeToGPUs; //TODO: is this needed? && other.M % M == 0;
  }

  // bool operator>(const SlicedMulShape& other) const {
  //   return KronCols > other.KronCols && KronRows > other.KronRows && ColsA > other.ColsA;
  // }

  friend std::ostream& operator<<(std::ostream &out, const SlicedMulShape &shape) {
    out << shape.P << "x" << shape.Q << "_" << shape.M << "x" << shape.K << "**" << shape.NumFusedKerns << "_" << shape.DistributeToGPUs;
    return out;
  }
};

template<>
struct std::hash<SlicedMulShape> {
  std::size_t operator()(const SlicedMulShape& k) const;
};

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