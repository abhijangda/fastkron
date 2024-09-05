#include <iostream>
#include <sstream>
#include <omp.h>

#include "kmm/matrix.h"
#include "utils/utils.h"
#include "handle/op.h"
#include "kernels/params.h"
#include "kernels/hw_details.h"

#pragma once

/**
 * KMMKernel is a CPU/GPU kernel to compute KMMProblem.
 * Each backend kernel is a subclass of KMMKernel.
 * This class stores the template parameters of a KMMKernel, while
 * the subclasses implements functions for invoking the kernel 
 */
struct KMMKernel {
  /**
   * @kernelInvoker: A function that invokes the kernel.
   * The CUDA/HIP function is of type:
   * void (*) (KernelParams<fusedFacs>, FusedParams<fusedFacs>, DistributedParams, EpilogueParams, dim3, dim3, uint32_t, cudaStream_t)
   * The CPU function is of type:
   * void (*) (KernelParams<fusedFacs>, FusedParams<fusedFacs>, DistributedParams, EpilogueParams)
   */
  void* kernelInvoker;

  /***Following fields store template parameter values of a kernel***/
  /**
   * @elemType: Element type of computation.
   */
  FastKronType elemType;

  /**
   * @f: Maximum factor size of the kernel.
   */
  Factor f;

  /**
   * @tileF: Values of tile size of P and Q.
   */
  Factor tileF;

  /**
   * @tileX: Values of tile size of M and columns of X (K).
   */
  Matrix tileX;

  /**
   * @fusedFacs: Number of fused factors of kernel.
   */
  uint fusedFacs;

  /**
   * @P2PStore: True if kernel uses P2P RDMA stores to write output.
   */
  bool P2PStore;

  /**
   * @regM, @regK, @regQ: Register tile values of M, K (cols of X), and Q.
   */
  uint regM; uint regK; uint regQ;

  /**
   * @optLevel: Optimization Level from 0 to 3 (see kernel_opt_levels.hpp).
   */
  uint optLevel;

  /**
   * @opX: fastKronOp of X.
   */
  fastKronOp opX;

  /**
   * @opF: fastKronOp of F.
   */
  fastKronOp opF;
  
  uint getFusedFacs() {return fusedFacs;}
  uint getOptLevel()  {return optLevel;}
  uint getRegM()      {return regM;}
  uint getRegK()      {return regK;}
  uint getRegQ()      {return regQ;}

  KMMKernel() {}

  //TODO: Order of template and this constructor must be same
  KMMKernel(void* kernelInvoker, Factor f, Factor tileF, Matrix tileX,
             uint fusedFacs, bool P2PStore,
             uint regM, uint regK, uint regQ, FastKronType elemType, uint optLevel,
             fastKronOp opX, fastKronOp opF) :
             kernelInvoker(kernelInvoker), elemType(elemType), f(f), tileF(tileF), tileX(tileX),
             fusedFacs(fusedFacs), P2PStore(P2PStore), regM(regM), regK(regK), regQ(regQ), optLevel(optLevel),
             opX(opX), opF(opF) {}
  bool isValid() {return kernelInvoker != nullptr;}
  virtual bool canCompute(KMMProblem problem, HardwareDetails*, bool p2p, bool exactFuse = true) {
    using Opts = KernelOptimizations::Optimization;

    bool ret = problem.type() == elemType &&
               problem.opFs() == opF && problem.opX() == opX && 
               P2PStore == p2p && ((exactFuse && problem.n() == fusedFacs) || (!exactFuse && problem.n() >= fusedFacs)) &&
               tileX.n()/problem.f(0).p() > 0; //Kernel's TileX is greater than P

    if (!ret) return false;

    bool followsAllOpts = true;
    uint lg = 0;
    for (Opts opt = Opts(lg); opt < Opts::NumOptimizations; opt = Opts(1 << lg), ++lg) {
      if ((KernelOptimizations::getOptimizations(optLevel) & opt) == opt) {
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
    info << strOfFastKronType(elemType) << "_" << f << "_" << tileF <<"_" << fusedFacs << "_" << tileX << "_" <<
            regM << "x" << regK << "x" << regQ << "_" << opX << opF << "_" << P2PStore << "_" << optLevel;
    return info.str();
  }
};

struct CPUKernel : public KMMKernel {
  CPUKernel() {}
  CPUKernel(void* kernelInvoker, Factor f, Factor tileF, Matrix tileX, 
            uint fusedFacs, bool P2PStore, 
            uint regM, uint regK, uint regQ, FastKronType elemType, uint optLevel,
            fastKronOp opX, fastKronOp opF) : 
            KMMKernel (kernelInvoker, f, tileF, tileX, 
                        fusedFacs, P2PStore, regM, regK, regQ, elemType, optLevel, opX, opF) {}
};

struct X86Kernel : public CPUKernel {
  X86SIMD simd;
  X86Kernel() {}
  X86Kernel(X86SIMD simd, void* kernelInvoker, Factor f, Factor tileF, Matrix tileX, 
            uint fusedFacs, bool P2PStore, 
            uint regM, uint regK, uint regQ, FastKronType elemType, uint optLevel,
            fastKronOp opX, fastKronOp opF) :
            CPUKernel(kernelInvoker, f, tileF, tileX, fusedFacs, P2PStore, regM, regK, regQ, elemType, optLevel, opX, opF),
            simd(simd) {}
  
  virtual std::string runtimeStr() const {
    return "X86";
  }

  virtual std::string archStr() const {
    return x86simdToStr(simd);
  }

  virtual std::string str() const {
    std::stringstream info;
    info << runtimeStr() << "_" << archStr() << "_" << KMMKernel::str();
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
  KMMKernel* kernel;
  uint start, end;
  uint K;
  float time;
  bool distShare;

  TunedKernelFromStart() {}
  TunedKernelFromStart(KMMKernel* kernel_, uint start_, uint end_, uint K_, float time_):
    kernel(kernel_), start(start_), end(end_), K(K_), time(time_), distShare(false) {}

  friend std::ostream& operator<<(std::ostream &out, const TunedKernelFromStart &k) {
    out << "[" << k.start << ", " << k.end << "] = " << k.K << " " 
        << k.kernel->str() << " runs for " << k.time << " ms";
    return out;
  }
};

typedef std::vector<TunedKernelFromStart> TunedKernelsSeries;