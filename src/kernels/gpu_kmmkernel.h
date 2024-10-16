#include <cuda.h>
#include <cuda_runtime.h>

#include "kmmkernel.h"

#pragma once

/**
 * GPUKMMKernel - A subclass for kernels running on GPUs.
 *                This class must be subclassed for CUDA or HIP kernels.
 */
struct GPUKMMKernel : public KMMKernel {
protected:
  /**
   * @kernel: Pointer to the kernel.
   */
  void* kernel;

  /**
   * @numThreads: Number of threads per threadblock.
   */
  uint numThreads;

  /**
   * @alignX: Alignment of pointer of X.
   */
  uint alignX;
  
  /**
   * @alignF: Alignment of pointer of F.
   */
  uint alignF;

public:
  GPUKMMKernel() {}
  GPUKMMKernel(void* kernelInvoker, FastKronType elemType,
               Factor f, Factor tileF, Matrix tileX, uint fusedFacs, bool P2PStore,
               uint regM, uint regK, uint regQ, uint optLevel,
               fastKronOp opX, fastKronOp opF, KernelBatchType::Ty kernelBatchType,
               void*(*getKernel)(), uint NumThreads,
               uint alignX, uint alignF) :
               KMMKernel(kernelInvoker, elemType, f, tileF, tileX,
                         fusedFacs, P2PStore, regM, regK, regQ,
                         optLevel, opX, opF, kernelBatchType),
               numThreads(NumThreads), kernel(getKernel()),
               alignX(alignX), alignF(alignF) {}

  /**
   * Getter for members
   */
  void* getKernel()     const {return kernel;}
  uint  getNumThreads() const {return numThreads;}
  uint  getAlignmentX() const {return alignX;}
  uint  getAlignmentF() const {return alignF;}

  /**
   * grid() - Returns grid size of the kernel for a problem. 
   */
  template<typename KMMProblem>
  dim3 grid(const KMMProblem& problem, int batchCount) const;

  template<uint32_t MaxFactors>
  dim3 grid(const KMMProblemT<MaxFactors>& problem) const;

  template<uint32_t MaxFactors>
  dim3 grid(const KMMProblemStridedBatchedT<MaxFactors>& problem) const;

  /**
   * block() - Returns block size of the kernel for a problem.
   */
  dim3 block() const {
    return dim3{getNumThreads(), 1, 1};
  }

  /**
   * getNumBlocks() - Returns number of blocks of the kernel for a problem.
   */
  uint32_t getNumBlocks(KMMProblem problem) const {
    dim3 g = grid(problem);
    return g.x*g.y*g.z;
  }
  uint32_t getNumBlocks(KMMProblemStridedBatched problem) const {
    dim3 g = grid(problem);
    return g.x*g.y*g.z;
  }

  /**
   * getMaxSharedMemSize() - Returns the maximum shared memory size of the kernel.
   *                         Effectively this is the maximum total tile size
   */
  size_t getMaxSharedMemSize() const {
    return getMaxTotalTileSize();
  }

  /**
   * getSharedMemSize() - Returns the shared memory size for the kernel.
   */
  size_t getSharedMemSize(KMMProblem problem) const {
    //TODO: Shouldn't this be MIN? because getTotalTileSize < getMaxTotalTileSize
    //TODO: Padding for Factor when OpY = fastKronOp_T with size 32, 128 for float
    return MAX(getTotalTileSize(problem), getMaxTotalTileSize()) + 32 * 4;
  }
  size_t getSharedMemSize(KMMProblemStridedBatched problem) const {
    return getSharedMemSize(problem.batchProblem(0));
  }

  /**
   * canCompute() - Overriding the method of KMMKernel.
   */
  virtual bool canCompute(KMMProblem problem, const HardwareDetails* hw, bool p2p,
                          KernelBatchType::Ty probBatchType, bool exactFuse = true);

  /**
   * str() - Overriding the method of KMMKernel. Adds NumThreads extra to the kernel string.
   */
  virtual std::string str() const;
};

template<typename KMMProblem>
dim3 GPUKMMKernel::grid(const KMMProblem& problem, int batchCount) const {
  Matrix tileX = getTileX(problem);
  Factor tileF = getTileF(problem);
  bool isNOrT = true; //true for N and false for T
  if (problem.mmtype() == FastKronMMType::MKM) {
    isNOrT = (problem.opX() == fastKronOp_N);
  } else {
    isNOrT = (problem.opX() == fastKronOp_T);
  }

  if (isNOrT) {
    return dim3(DIVUP(problem.k(), tileX.n()) * DIVUP(problem.f(0).q(), tileF.q()),
                DIVUP(problem.m(), tileX.m()),
                batchCount);
  } else {
    //problem.k() can be very large, which can make grid.y more than the limit (65535).
    //Distribute grid.y to y and z.
    uint32_t origGridy = DIVUP(problem.k(), tileX.n()) *
                         DIVUP(problem.f(0).q(), tileF.q());
    dim3 grid = {0,0,0};
    if (origGridy <= 32768) {
      grid.y = origGridy;
      grid.z = batchCount;
    } else {
      //TODO: What if stridedbatched is used for large  grid y?
      grid.y = 32768;
      grid.z = DIVUP(origGridy, 32768);
    }
    return dim3(DIVUP(problem.m(), tileX.m()), grid.y, grid.z);
  }
}

template<uint32_t MaxFactors>
dim3 GPUKMMKernel::grid(const KMMProblemT<MaxFactors>& problem) const {
  return grid(problem, 1);
}

template<uint32_t MaxFactors>
dim3 GPUKMMKernel::grid(const KMMProblemStridedBatchedT<MaxFactors>& problem) const {
  return grid(problem.batchProblem(0), problem.batchCount());
}