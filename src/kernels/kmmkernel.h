#include <iostream>
#include <sstream>
#include <omp.h>

#include "kmm/matrix.h"
#include "utils/utils.h"
#include "handle/op.h"
#include "kernels/params.h"
#include "kernels/hw_details.h"

#pragma once

namespace KernelBatchType {
  enum Ty {
    Normal,
    StridedBatch,
    Batch
  };
}

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
protected:

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

  KernelBatchType::Ty kernelBatchType;

public:
  KMMKernel() {}

  KMMKernel(void* kernelInvoker, FastKronType elemType,
            Factor f, Factor tileF, Matrix tileX, uint fusedFacs, bool P2PStore,
            uint regM, uint regK, uint regQ, uint optLevel,
            fastKronOp opX, fastKronOp opF) :
            kernelInvoker(kernelInvoker), elemType(elemType),
            f(f), tileF(tileF), tileX(tileX), fusedFacs(fusedFacs),
            P2PStore(P2PStore), regM(regM), regK(regK), regQ(regQ),
            optLevel(optLevel), opX(opX), opF(opF) {}

  /**
   * Getters for all template parameters
   */
  uint getFusedFacs()   const {return fusedFacs;}
  uint getOptLevel()    const {return optLevel;}
  uint getRegM()        const {return regM;}
  uint getRegK()        const {return regK;}
  uint getRegQ()        const {return regQ;}
  Factor getMaxFactor() const {return f;}
  Matrix getMaxTileX()  const {return tileX;}
  Factor getMaxTileF()  const {return tileF;}
  fastKronOp getOpX()   const {return opX;}
  fastKronOp getOpF()   const {return opF;}

  /**
   * Check if a kernel is valid.
   */
  bool isValid()        const {return kernelInvoker != nullptr;}

  /**
   * getMaxTotalTileSize - Return max tile size as sum of tileF and tileX.
   */
  size_t getMaxTotalTileSize() const;

  /**
   * getMaxTileY - Return max size of Y tile written by the kernel.
   */
  Matrix getMaxTileY() const;

  /**
   * getTileF - Return tile of factor as a minimum of max factor tile and 
   *            problem's factor.
   */
  Factor getTileF(KMMProblem problem) const;
  Factor getTileF(KMMProblemStridedBatched problem) const {
    return getTileF(problem.batchProblem(0));
  }

  /**
   * getTileX - Return tile of X as a minimum of max slices of kernel and 
   *            slices of problem. 
   */
  Matrix getTileX(KMMProblem problem) const;
  Matrix getTileX(KMMProblemStridedBatched problem) const {
    return getTileX(problem.batchProblem(0));
  }

  /**
   * getTotalTileSize - Return the sum in bytes of tile sizes of getTileF and getTileX.
   */
  size_t getTotalTileSize(KMMProblem problem) const;
  size_t getTotalTileSize(KMMProblemStridedBatched problem) const {
    return getTotalTileSize(problem.batchProblem(0));
  }

  /**
   * getNumThreads - Return number of threads created by the kernel for problem.
   */
  size_t getNumThreads(KMMProblem problem) const;
  size_t getNumThreads(KMMProblemStridedBatched problem) const {
    return getNumThreads(problem.batchProblem(0)) * problem.batchCount();
  }

  /**
   * isOptValid - Return true if kernel and optimization is valid to compute for a problem. 
   */
  bool isOptValid(KMMProblem problem, KernelOptimizations::Optimization opt) const;

  /**
   * canCompute - Return true if a problem can be computed by the kernel on given hardware.
   *              This method can be overriden by subclasses.
   * @problem: KMM problem
   * @hw: Underlying hardware
   * @p2p: True if P2P stores are needed to shared output otherwise False
   * @exactFuse: True 
   */
  virtual bool canCompute(KMMProblem problem, const HardwareDetails* hw, 
                          bool p2p, bool exactFuse = true);
  bool canCompute(KMMProblemStridedBatched problem, const HardwareDetails* hw, 
                  bool p2p, bool exactFuse = true) {
    return kernelBatchType == KernelBatchType::StridedBatch &&
           canCompute(problem.batchProblem(0), hw, p2p, exactFuse);
  }
  /**
   * backend - Return backend (X86, CUDA, ARM, HIP) as string of the kernel.
   *           This method must be implemented by subclasses.
   */
  virtual std::string backend() const = 0;

  /**
   * arch - Return underlying architecture of the kernel.
   *        This method must be implemented by subclass.
   */
  virtual std::string arch() const = 0;

  /**
   * str - Return string representation of the kernel.
   */
  virtual std::string str() const;
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