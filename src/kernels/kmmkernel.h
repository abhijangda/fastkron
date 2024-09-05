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

  KMMKernel() {}

  KMMKernel(void* kernelInvoker, FastKronType elemType,
            Factor f, Factor tileF, Matrix tileX, uint fusedFacs, bool P2PStore,
            uint regM, uint regK, uint regQ, uint optLevel,
            fastKronOp opX, fastKronOp opF) :
            kernelInvoker(kernelInvoker), elemType(elemType),
            f(f), tileF(tileF), tileX(tileX), fusedFacs(fusedFacs),
            P2PStore(P2PStore), regM(regM), regK(regK), regQ(regQ),
            optLevel(optLevel), opX(opX), opF(opF) {}

  bool isValid()      const {return kernelInvoker != nullptr;}
  uint getFusedFacs() const {return fusedFacs;}
  uint getOptLevel()  const {return optLevel;}
  uint getRegM()      const {return regM;}
  uint getRegK()      const {return regK;}
  uint getRegQ()      const {return regQ;}

  size_t totalTileSize() const;

  Matrix getTileY() const;

  Factor getTileF(KMMProblem problem) const;

  Matrix getTileX(KMMProblem problem) const;

  size_t totalTileSize(KMMProblem problem) const;

  size_t numThreads(KMMProblem problem) const;

  virtual bool canCompute(KMMProblem problem, HardwareDetails*, bool p2p, bool exactFuse = true) const;

  bool validOptFor(KMMProblem problem, KernelOptimizations::Optimization opt) const;

  virtual std::string runtimeStr() const = 0;

  virtual std::string archStr() const = 0;

  virtual std::string str() const;
};

struct CPUKernel : public KMMKernel {
  CPUKernel() {}
  CPUKernel(void* kernelInvoker, FastKronType elemType, Factor f, Factor tileF, Matrix tileX, 
            uint fusedFacs, bool P2PStore, 
            uint regM, uint regK, uint regQ, uint optLevel,
            fastKronOp opX, fastKronOp opF) : 
            KMMKernel (kernelInvoker, elemType, f, tileF, tileX, 
                        fusedFacs, P2PStore, regM, regK, regQ, optLevel, opX, opF) {}
};

struct X86Kernel : public CPUKernel {
  X86SIMD simd;
  X86Kernel() {}
  X86Kernel(X86SIMD simd, void* kernelInvoker, FastKronType elemType, Factor f, Factor tileF, Matrix tileX, 
            uint fusedFacs, bool P2PStore, 
            uint regM, uint regK, uint regQ, uint optLevel,
            fastKronOp opX, fastKronOp opF) :
            CPUKernel(kernelInvoker, elemType, f, tileF, tileX, fusedFacs, P2PStore, regM, regK, regQ, optLevel, opX, opF),
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