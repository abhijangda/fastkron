#include <iostream>
#include <sstream>
#include <omp.h>

#include "kmm/matrix.h"
#include "utils/utils.h"
#include "handle/op.h"
#include "kernels/params.h"
#include "kernels/hw_details.h"

#pragma once

struct KernelInfo {
  void* invokerFunc;
  Factor f;
  Factor tileF;
  Matrix tileX;
  fastKronOp opX;
  fastKronOp opF;
  uint RegM;
  uint RegK;
  uint RegQ;
  uint FusedFacs;
  FastKronType elemType;
  uint OptLevel;
  bool DistributeToGPUs;
  
  KernelInfo() {}
  KernelInfo(void* invokerFunc, Factor f, Factor tileF, Matrix tileX,
             uint FusedFacs, bool DistributeToGPUs,
             uint RegM, uint RegK, uint RegQ, FastKronType elemType, uint OptLevel,
             fastKronOp opX, fastKronOp opF) :
             invokerFunc(invokerFunc), f(f), tileF(tileF), tileX(tileX),
             FusedFacs(FusedFacs), DistributeToGPUs(DistributeToGPUs),
             RegM(RegM), RegK(RegK), RegQ(RegQ), OptLevel(OptLevel), elemType(elemType),
             opX(opX), opF(opF) {}
  bool isValid() {return invokerFunc != nullptr;}
  virtual bool canCompute(KMMProblem problem, HardwareDetails* hardware, bool p2p, bool exactFuse = true) {
    using Opts = KernelOptimizations::Optimization;

    bool ret = problem.type() == elemType &&
               problem.opFs() == opF && problem.opX() == opX && 
               DistributeToGPUs == p2p && ((exactFuse && problem.n() == FusedFacs) || (!exactFuse && problem.n() >= FusedFacs)) &&
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
    //TODO: make this tileF.size() + Xsh.size()
    return (tileF.numel() + Xsh.numel())*sizeOfFastKronType(elemType);
  }

  Matrix getTileY() {
    return Matrix(tileX.m(), (tileX.n()/f.p()) * tileF.q());
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
    return (tileF.numel() + Xsh.numel())*sizeOfFastKronType(elemType);
  }

  size_t numThreads(KMMProblem problem) {
    Matrix tileX_ = getTileX(problem);
    Factor tileF_ = getTileF(problem);

    return DIVUP(problem.k(), tileX_.n()) * 
           DIVUP(problem.f(0).q(), tileF_.q()) * 
           DIVUP(problem.m(), tileX_.m());
  }

  bool validOptFor(KMMProblem problem, KernelOptimizations::Optimization opt);

  virtual std::string runtimeStr() const {
    assert (false);
    return "";
  }

  virtual std::string archStr() const {
    assert(false);
    return "";
  }

  virtual std::string str() const {
    std::stringstream info;
    info << strOfFastKronType(elemType) << "_" << f << "_" << tileF <<"_" << FusedFacs << "_" << tileX << "_" <<
            RegM << "x" << RegK << "x" << RegQ << "_" << opX << opF << "_" << DistributeToGPUs << "_" << OptLevel;
    return info.str();
  }
};

struct CPUKernel : public KernelInfo {
  CPUKernel() {}
  CPUKernel(void* invokerFunc, Factor f, Factor tileF, Matrix tileX, 
            uint FusedFacs, bool DistributeToGPUs, 
            uint RegM, uint RegK, uint RegQ, FastKronType elemType, uint OptLevel,
            fastKronOp opX, fastKronOp opF) : 
            KernelInfo (invokerFunc, f, tileF, tileX, 
                        FusedFacs, DistributeToGPUs, RegM, RegK, RegQ, elemType, OptLevel, opX, opF) {}
};

struct X86Kernel : public CPUKernel {
  X86SIMD simd;
  X86Kernel() {}
  X86Kernel(X86SIMD simd, void* invokerFunc, Factor f, Factor tileF, Matrix tileX, 
            uint FusedFacs, bool DistributeToGPUs, 
            uint RegM, uint RegK, uint RegQ, FastKronType elemType, uint OptLevel,
            fastKronOp opX, fastKronOp opF) :
            simd(simd), CPUKernel(invokerFunc, f, tileF, tileX, FusedFacs, DistributeToGPUs, RegM, RegK, RegQ, elemType, OptLevel, opX, opF) {}
  
  virtual std::string runtimeStr() const {
    return "X86";
  }

  virtual std::string archStr() const {
    return x86simdToStr(simd);
  }

  virtual std::string str() const {
    std::stringstream info;
    info << runtimeStr() << "_" << archStr() << "_" << KernelInfo::str();
    return info.str();
  }

  virtual bool canCompute(KMMProblem problem, HardwareDetails* hardware, bool p2p, bool exactFuse = true) {
    if (CPUKernel::canCompute(problem, hardware, p2p, exactFuse)) {
      return simd <= ((X86ArchDetails*)hardware)->simd;
    }
    return false;
  }
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