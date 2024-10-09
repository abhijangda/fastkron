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
               fastKronOp opX, fastKronOp opF, KernelBatchType kernelBatchType,
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
  dim3 grid(KMMProblem problem) const;

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
    return MAX(getTotalTileSize(problem), getMaxTotalTileSize());
  }

  /**
   * canCompute() - Overriding the method of KMMKernel.
   */
  virtual bool canCompute(KMMProblem problem, const HardwareDetails* hw, bool p2p, 
                          bool exactFuse = true);

  /**
   * str() - Overriding the method of KMMKernel. Adds NumThreads extra to the kernel string.
   */
  virtual std::string str() const;
};