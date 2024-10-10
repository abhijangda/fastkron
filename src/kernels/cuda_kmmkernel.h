#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels/gpu_kmmkernel.h"

#pragma once

struct CUDAKMMKernel : public GPUKMMKernel {
  SMArch sm;
  CUDAKMMKernel() {}
  CUDAKMMKernel(SMArch sm, void* kernelInvoker, FastKronType elemType,
               Factor f, Factor tileF, Matrix tileX, uint fusedFacs, bool P2PStore,
               uint regM, uint regK, uint regQ, uint optLevel,
               fastKronOp opX, fastKronOp opF,
               void*(*getKernel)(), uint NumThreads,
               uint alignX, uint alignF) :
               GPUKMMKernel(kernelInvoker, elemType, f, tileF, tileX,
                            fusedFacs, P2PStore, regM, regK, regQ,
                            optLevel, opX, opF, getKernel, 
                            NumThreads, alignX, alignF),
               sm(sm) {}

  /*** Functions to get/set information for CUDA Kernel ***/
  /**
   * getPTXVersion() - Return PTX Version of the kernel as XXYY.
   */
  uint32_t getPTXVersion() const {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    return attr.ptxVersion;
  }

  /**
   * getLocalSize() - Return local memory size in bytes.
   */
  uint32_t getLocalSize() const {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    return attr.localSizeBytes;
  }
  
  /**
   * getNumRegs() - Return number of registers per thread.
   */
  uint32_t getNumRegs()   const {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    return attr.numRegs;
  }

  /**
   * setSharedMemAttr() - Set MaxDynamicSharedMemorySize attribute of the kernel 
   *                      if shared memory is more than 48KB.
   */
  cudaError_t setSharedMemAttr() {
    cudaError_t err = cudaSuccess;
    if (getMaxSharedMemSize() >= (48 << 10)) {
      err = cudaFuncSetAttribute(kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 getMaxSharedMemSize());
    }

    return err;
  }
  /*********************************************************/

  /**
   * canCompute() - Overrides method of GPUKMMKernel
   */
  virtual bool canCompute(KMMProblem problem, const HardwareDetails* hw,
                          bool p2p, KernelBatchType::Ty probBatchType, bool exactFuse = true) {
    if (GPUKMMKernel::canCompute(problem, hw, p2p, probBatchType, exactFuse)) {
      return ((CUDAArchDetails*)hw)->smArch == sm;
    }
    return false;
  }

  /**
   * backend() - Returns CUDA as backend.
   */
  virtual std::string backend() const {
    return "cuda";
  }

  /**
   * arch() - Returns SM string.
   */
  virtual std::string arch() const {
    return smArchToStr(sm);
  }
};