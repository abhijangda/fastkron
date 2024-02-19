#include <iostream>
#include <sstream>

#include "kmm/matrix.h"
#include "utils/utils.h"
#include "handle/op.h"

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
  
  Factor factor;
  Factor tiledFactor;
  Matrix tiledInput;

  fastKronOp opX;
  fastKronOp opF;
  
  uint NumFusedKerns_;
  ElementType elemType;

  bool DistributeToGPUs_;
 
  KernelInfo() {}
  KernelInfo(void* invokerFunc, void*(*getKernelFunc)(), 
             uint Q, uint P, uint tileP, uint tileQ,
             uint TileK, uint TileM, uint NumFusedKerns_, bool DistributeToGPUs_, ElementType elemType_,
             fastKronOp opX, fastKronOp opF) :
             invokerFunc(invokerFunc), kernelFunc(getKernelFunc()), factor(P, Q), tiledFactor(tileP, tileQ),
             tiledInput(TileM, TileK), NumFusedKerns_(NumFusedKerns_), DistributeToGPUs_(DistributeToGPUs_),
             elemType(elemType_), opX(opX), opF(opF) {}
  bool canCompute(KMMProblem problem, bool p2p) {
    return Factor(factor.p(), tiledFactor.q()) == problem.f(0) &&
           problem.opFs() == opF &&
           problem.opX()  == opX &&
           problem.k() % tiledInput.n() == 0 &&
           problem.n() == NumFusedKerns_ &&
           DistributeToGPUs_ == p2p;
  }
  virtual std::string str() const = 0;
};

struct CPUKernel : public KernelInfo {
  CPUKernel() {}
  CPUKernel(void* invokerFunc, void*(*getKernelFunc)(), 
             uint Q, uint P, uint tileP, uint tileQ,
             uint TileK, uint TileM, uint NumFusedKerns_, bool DistributeToGPUs_, ElementType elemType_,
             fastKronOp opX, fastKronOp opF) : 
             KernelInfo (invokerFunc, getKernelFunc, Q, P, tileP, tileQ, TileK, TileM, 
                         NumFusedKerns_, DistributeToGPUs_, elemType_, opX, opF) {}
  std::string str() const {
    std::stringstream info;
    info << tiledFactor << "_" << tiledInput << "**" << NumFusedKerns_ << "_" << DistributeToGPUs_ << "_" << opX << opF;
    return info.str();
  } 
};

struct CUDAKernel : public KernelInfo {
  uint NumThreads;
  
  uint CRegRows;
  uint CRegCols;
  ElementType elemType;
  uint AAlignment;
  uint KronAlignment;
  CUDAKernel() {}
  CUDAKernel(void* invokerFunc, void*(*getKernelFunc)(), uint NumThreads_, 
             uint Q, uint P, uint tileP, uint tileQ,
             uint TileK, uint TileM, uint NumFusedKerns_, bool DistributeToGPUs_, 
             uint CRegRows_, uint CRegCols_, ElementType elemType_,
             uint AAlignment_, uint KronAlignment_,
             fastKronOp opX, fastKronOp opF) :
             KernelInfo(invokerFunc, getKernelFunc, Q, P, tileP, tileQ, TileK, TileM, NumFusedKerns_, DistributeToGPUs_, elemType_, opX, opF),
             NumThreads(NumThreads_),
             CRegRows(CRegRows_), CRegCols(CRegCols_),
             AAlignment(AAlignment_), KronAlignment(KronAlignment_) {}

  bool isValid() {return invokerFunc != nullptr && kernelFunc != nullptr;}

  std::string str() const {
    std::stringstream info;
    info << NumThreads << "_" << tiledFactor << "_" << tiledInput << "**" << 
          NumFusedKerns_ << "_"<< DistributeToGPUs_ << "_" <<
          CRegRows << "x" << CRegCols << "_" <<
          AAlignment << "_" << KronAlignment << "_" << opX << opF;
    return info.str();
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