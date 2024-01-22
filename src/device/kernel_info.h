#include <iostream>

#include "kmm/matrix.h"
#include "utils/utils.h"

#pragma once

enum ElementType {
  Float,
  Double,
  Int,
  Long
};

struct KernelInfo {
  void* invokerFunc;
  void* kernelFunc;
  uint NumThreads;
  
  Factor factor;
  Factor tiledFactor;
  Matrix tiledInput;

  fastKronOp opX;
  fastKronOp opF;
  
  uint NumFusedKerns_;
  bool DistributeToGPUs_;
  
  uint CRegRows;
  uint CRegCols;
  ElementType elemType;
  uint AAlignment;
  uint KronAlignment;
  KernelInfo() {}
  KernelInfo(void* invokerFunc, void*(*getKernelFunc)(), uint NumThreads_, 
             uint Q, uint P, uint tileP, uint tileQ,
             uint TileK, uint TileM, uint NumFusedKerns_, bool DistributeToGPUs_, 
             uint CRegRows_, uint CRegCols_, ElementType elemType_,
             uint AAlignment_, uint KronAlignment_,
             fastKronOp opX, fastKronOp opF) :
             invokerFunc(invokerFunc), kernelFunc(getKernelFunc()), NumThreads(NumThreads_), factor(P, Q), tiledFactor(tileP, tileQ),
             tiledInput(TileM, TileK), NumFusedKerns_(NumFusedKerns_), DistributeToGPUs_(DistributeToGPUs_),
             CRegRows(CRegRows_),
             CRegCols(CRegCols_), elemType(elemType_),
             AAlignment(AAlignment_), KronAlignment(KronAlignment_),
             opX(opX), opF(opF) {}

  bool isValid() {return invokerFunc != nullptr && kernelFunc != nullptr;}
  friend std::ostream& operator<<(std::ostream &out, const KernelInfo &info) {
    out << info.NumThreads << "_" << info.tiledFactor << "_" << info.tiledInput << "**" << 
          info.NumFusedKerns_ << "_"<< info.DistributeToGPUs_
          << "_" <<
           info.CRegRows << "x" << info.CRegCols << "_" <<
           info.AAlignment << "_" << info.KronAlignment << "_" << info.opX << info.opF;
      
    return out;
  }

  bool canCompute(KMMProblem problem, bool p2p) {
    return Factor(factor.p(), tiledFactor.q()) == problem.f(0) &&
           problem.opFs() == opF &&
           problem.opX()  == opX &&
           problem.k() % tiledInput.n() == 0 &&
           problem.n() == NumFusedKerns_ &&
           DistributeToGPUs_ == p2p;
  }

  dim3 grid(KMMProblem problem) {
    return dim3 {
                  problem.k()/tiledInput.n() * DIVUP(problem.f(0).q(), tiledFactor.q()),
                  DIVUP(problem.m(), tiledInput.m()),
                  1
                };
  }

  dim3 block() {
    return dim3{NumThreads, 1, 1};
  }

  size_t sharedMemSize() {
    Matrix Xsh = Matrix(tiledInput.m(), (tiledInput.n()/factor.p())*tiledFactor.p());
    return (tiledFactor.numel() + Xsh.numel())*sizeof(float);
  }

  cudaError_t setSharedMemAttr() {
    cudaError_t err = cudaSuccess;
    if (sharedMemSize() >= (48 << 10)) {
      err = cudaFuncSetAttribute(kernelFunc,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 sharedMemSize());
    }

    return err;
  }
};