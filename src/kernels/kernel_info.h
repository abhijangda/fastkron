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
  Factor f;
  Factor tileF;
  Matrix tileX;
  fastKronOp opX;
  fastKronOp opF;
  uint RegK;
  uint RegQ;
  uint FusedFacs;
  ElementType elemType;
  bool DistributeToGPUs;
  
  KernelInfo() {}
  KernelInfo(void* invokerFunc, Factor f, Factor tileF, Matrix tileX,
             uint FusedFacs, bool DistributeToGPUs,
             uint RegK, uint RegQ, ElementType elemType,
             fastKronOp opX, fastKronOp opF) :
             invokerFunc(invokerFunc), f(f), tileF(tileF), tileX(tileX),
             FusedFacs(FusedFacs), DistributeToGPUs(DistributeToGPUs),
             RegK(RegK), RegQ(RegQ), elemType(elemType), opX(opX), opF(opF) {}
  bool isValid() {return invokerFunc != nullptr;}
  bool canCompute(KMMProblem problem, bool p2p) {
    return f == problem.f(0) && 
           problem.f(0).q() % tileF.q() == 0 &&
           problem.opFs() == opF &&
           problem.opX()  == opX &&
           problem.k() % tileX.n() == 0 &&
           problem.n() == FusedFacs &&
           DistributeToGPUs == p2p;
  }

  size_t totalTileSize() {
    Matrix Xsh = Matrix(tileX.m(), (tileX.n()/f.p())*tileF.p());
    return (tileF.numel() + Xsh.numel())*sizeof(float);
  }

  virtual std::string str() const {
    std::stringstream info;
    info << tileF << "_" << tileX << "**" << FusedFacs << "_" << 
            DistributeToGPUs << "_" << RegK << "x" << RegQ << "_" << opX << opF;
    return info.str();
  }
};

struct CPUKernel : public KernelInfo {
  CPUKernel() {}
  CPUKernel(void* invokerFunc, Factor f, Factor tileF, Matrix tileX, 
            uint FusedFacs, bool DistributeToGPUs, 
            uint RegK, uint RegQ, ElementType elemType,
            fastKronOp opX, fastKronOp opF) : 
            KernelInfo (invokerFunc, f, tileF, tileX, 
                        FusedFacs, DistributeToGPUs, RegK, RegQ, elemType, opX, opF) {} 
};