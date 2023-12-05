#include <iostream>

#include "sliced_mul_shape.h"

#pragma once

enum ElementType {
  Float,
  Double,
  Int,
  Long
};

struct KernelInfo {
  void* kernel;
  uint NumThreads;
  
  uint Q;
  SlicedMulShape tiledShape;
  
  uint CRegRows;
  uint CRegCols;
  ElementType elemType;
  bool RowModTileIsZero;
  bool KEqVar;
  uint AAlignment;
  uint KronAlignment;

  KernelInfo() : kernel(nullptr) {}
  KernelInfo(void* kernel_, uint NumThreads_, uint Q, uint P, uint tileQ,
             uint TileK, uint TileM, uint NumFusedKerns_, bool DistributeToGPUs_, 
             uint CRegRows_, uint CRegCols_, ElementType elemType_, bool RowModTileIsZero_, bool KEqVar_,
             uint AAlignment_, uint KronAlignment_) :
             kernel(kernel_), NumThreads(NumThreads_), Q(Q), tiledShape{tileQ, P, TileK, TileM, NumFusedKerns_, DistributeToGPUs_},
             CRegRows(CRegRows_),
             CRegCols(CRegCols_), elemType(elemType_), 
             RowModTileIsZero(RowModTileIsZero_), KEqVar(KEqVar_),
             AAlignment(AAlignment_), KronAlignment(KronAlignment_) {}

  bool isValid() {return kernel != nullptr;}
  friend std::ostream& operator<<(std::ostream &out, const KernelInfo &info) {
    out << info.NumThreads << "_" << info.tiledShape << "_" <<
           info.CRegRows << "x" << info.CRegCols << "_" <<
           info.RowModTileIsZero << "_" << 
           info.KEqVar << "_" << 
           info.AAlignment << "_" << info.KronAlignment;
      
    return out;
  }

  bool canCompute(SlicedMulShape shape) {
    return RowModTileIsZero == ((shape.M % tiledShape.M) == 0) &&
           tiledShape.isTileOf(shape); 
  }
};