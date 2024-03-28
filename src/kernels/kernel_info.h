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
  bool MaxShapeEq;
  bool DistributeToGPUs;
  
  KernelInfo() {}
  KernelInfo(void* invokerFunc, Factor f, Factor tileF, Matrix tileX,
             uint FusedFacs, bool DistributeToGPUs,
             uint RegK, uint RegQ, ElementType elemType, bool MaxShapeEq,
             fastKronOp opX, fastKronOp opF) :
             invokerFunc(invokerFunc), f(f), tileF(tileF), tileX(tileX),
             FusedFacs(FusedFacs), DistributeToGPUs(DistributeToGPUs),
             RegK(RegK), RegQ(RegQ), MaxShapeEq(MaxShapeEq), elemType(elemType), opX(opX), opF(opF) {}
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

  Factor getTileF(KMMProblem problem) {
    Factor f_ = problem.f(0);
    return Factor(MIN(tileF.p(), f_.p()), MIN(tileF.q(), f_.q()));
  }

  Matrix getTileX(KMMProblem problem) {
    Factor f_ = problem.f(0);
    Factor tileF_ = getTileF(problem);

    uint32_t kernelTileSlices = tileX.n()/f.p();
    uint32_t problemTileSlices = problem.x().n()/f_.p();
    uint32_t slices;
    if (problemTileSlices >= kernelTileSlices) {
      slices = kernelTileSlices;
    } else {
      slices = MIN(tileX.n()/f_.p(), kernelTileSlices);
      slices = MIN(problemTileSlices, slices);
    }
    return Matrix(tileX.m(), slices * f_.p());
  }

  size_t totalTileSize(KMMProblem problem) {
    Matrix tileX_ = getTileX(problem);
    Factor tileF_ = getTileF(problem);
    Factor f_ = problem.f(0);

    //Pad Xsh to TileP
    //Pad Fsh to TileP x TileQ
    Matrix Xsh = Matrix(tileX_.m(), 
                        (tileX_.n()/f_.p()) * tileF.p());
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
            uint RegK, uint RegQ, ElementType elemType, bool MaxShapeEq,
            fastKronOp opX, fastKronOp opF) : 
            KernelInfo (invokerFunc, f, tileF, tileX, 
                        FusedFacs, DistributeToGPUs, RegK, RegQ, elemType, MaxShapeEq, opX, opF) {} 
};