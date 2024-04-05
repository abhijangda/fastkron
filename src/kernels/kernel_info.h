#include <iostream>
#include <sstream>

#include "kmm/matrix.h"
#include "utils/utils.h"
#include "handle/op.h"
#include "kernels/params.h"

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
  uint OptLevel;
  bool DistributeToGPUs;
  
  KernelInfo() {}
  KernelInfo(void* invokerFunc, Factor f, Factor tileF, Matrix tileX,
             uint FusedFacs, bool DistributeToGPUs,
             uint RegK, uint RegQ, ElementType elemType, uint OptLevel,
             fastKronOp opX, fastKronOp opF) :
             invokerFunc(invokerFunc), f(f), tileF(tileF), tileX(tileX),
             FusedFacs(FusedFacs), DistributeToGPUs(DistributeToGPUs),
             RegK(RegK), RegQ(RegQ), OptLevel(OptLevel), elemType(elemType),
             opX(opX), opF(opF) {}
  bool isValid() {return invokerFunc != nullptr;}
  bool canCompute(KMMProblem problem, bool p2p) {
    using Opts = KernelOptimizations::Optimization;

    bool ret = problem.opFs() == opF && problem.opX() == opX && 
               DistributeToGPUs == p2p && problem.n() == FusedFacs &&
               tileX.n()/problem.f(0).p() > 0; //Kernel's TileX is greater than P

    if (!ret) return false;

    bool followsAllOpts = true;
    uint lg = 0;
    for (Opts opt = Opts(lg); opt < Opts::NumOptimizations; opt = Opts(1 << lg), ++lg) {
      if ((KernelOptimizations::getOptimizations(OptLevel) & opt) == opt) {
        followsAllOpts = followsAllOpts && validOptFor(problem, opt);
    }}

    return followsAllOpts;
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

    uint32_t slices = 0;
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

  bool validOptFor(KMMProblem problem, KernelOptimizations::Optimization opt);

  virtual std::string str() const {
    std::stringstream info;
    info << f << "_" << tileF <<"_" << FusedFacs << "_" << tileX << "_" <<
            RegK << "x" << RegQ << "_" << opX << opF << "_" << DistributeToGPUs << "_" << OptLevel;
    return info.str();
  }
};

struct CPUKernel : public KernelInfo {
  CPUKernel() {}
  CPUKernel(void* invokerFunc, Factor f, Factor tileF, Matrix tileX, 
            uint FusedFacs, bool DistributeToGPUs, 
            uint RegK, uint RegQ, ElementType elemType, uint OptLevel,
            fastKronOp opX, fastKronOp opF) : 
            KernelInfo (invokerFunc, f, tileF, tileX, 
                        FusedFacs, DistributeToGPUs, RegK, RegQ, elemType, OptLevel, opX, opF) {} 
};

#include <vector>

struct TunedKernelFromStart {
  //TODO: Cannot improve unless distributed code is refactored 
  KernelInfo* kernel;
  uint start, end;
  uint K;
  float time;
  bool distShare;

  TunedKernelFromStart() {}
  TunedKernelFromStart(KernelInfo* kernel_, uint start_, uint end_, uint K_, float time_):
    kernel(kernel_), start(start_), end(end_), K(K_), time(time_), distShare(false) {}

  friend std::ostream& operator<<(std::ostream &out, const TunedKernelFromStart &k) {
    out << "[" << k.start << ", " << k.end << "] = " << k.K << " " 
        << k.kernel->str() << " runs for " << k.time << " ms";
    return out;
  }
};

typedef std::vector<TunedKernelFromStart> TunedKernelsSeries;