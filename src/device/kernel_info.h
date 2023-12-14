#include <iostream>

#include "sliced_mul_shape.h"
#include "utils/utils.h"

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
  
  Factor factor;
  Factor tiledFactor;
  Matrix tiledInput;
  
  uint NumFusedKerns_;
  bool DistributeToGPUs_;
  
  uint CRegRows;
  uint CRegCols;
  ElementType elemType;
  bool RowModTileIsZero;
  bool KEqVar;
  uint AAlignment;
  uint KronAlignment;
  KernelInfo() {}
  KernelInfo(void* kernel_, uint NumThreads_, uint Q, uint P, uint tileQ,
             uint TileK, uint TileM, uint NumFusedKerns_, bool DistributeToGPUs_, 
             uint CRegRows_, uint CRegCols_, ElementType elemType_, bool RowModTileIsZero_, bool KEqVar_,
             uint AAlignment_, uint KronAlignment_) :
             kernel(kernel_), NumThreads(NumThreads_), factor(Q, P), tiledFactor(tileQ, P),
             tiledInput(TileM, TileK), NumFusedKerns_(NumFusedKerns_), DistributeToGPUs_(DistributeToGPUs_),
             CRegRows(CRegRows_),
             CRegCols(CRegCols_), elemType(elemType_), 
             RowModTileIsZero(RowModTileIsZero_), KEqVar(KEqVar_),
             AAlignment(AAlignment_), KronAlignment(KronAlignment_) {}

  bool isValid() {return kernel != nullptr;}
  friend std::ostream& operator<<(std::ostream &out, const KernelInfo &info) {
    out << info.NumThreads << "_" << info.tiledFactor << "_" << info.tiledInput << "**" << 
          info.NumFusedKerns_ << "_"<< info.DistributeToGPUs_
          << "_" <<
           info.CRegRows << "x" << info.CRegCols << "_" <<
           info.RowModTileIsZero << "_" << 
           info.KEqVar << "_" << 
           info.AAlignment << "_" << info.KronAlignment;
      
    return out;
  }

  bool canCompute(KMMProblem problem, bool p2p) {
    return RowModTileIsZero == ((problem.m % tiledInput.M) == 0) &&
           tiledFactor == Factor(problem.qs[0], problem.ps[0]) &&
           problem.k % tiledInput.N == 0 &&
           problem.n == NumFusedKerns_ &&
           DistributeToGPUs_ == p2p;
  }

  dim3 grid(KMMProblem problem) {
    return dim3 {
                  problem.k/tiledInput.N * DIVUP(problem.qs[0], tiledFactor.Q),
                  DIVUP(problem.m, tiledInput.M),
                  1
                };
  }

  dim3 block() {
    return dim3{NumThreads, 1, 1};
  }
};