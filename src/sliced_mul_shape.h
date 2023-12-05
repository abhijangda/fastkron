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