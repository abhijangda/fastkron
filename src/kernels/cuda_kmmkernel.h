#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels/gpu_kmmkernel.h"

#pragma once

struct CUDAKernel : public GPUKMMKernel {
  SMArch sm;
  CUDAKernel() {}
  CUDAKernel(SMArch sm, void* invokerFunc, FastKronType elemType, Factor f, Factor tileF, Matrix tileX, 
             uint FusedFacs, bool DistributeToGPUs,
             uint RegM, uint RegK, uint RegQ, uint OptLevel,
             fastKronOp opX, fastKronOp opF,
             void*(*getKernelFunc)(), uint NumThreads,
             uint AAlignment, uint KronAlignment) :
             GPUKMMKernel(invokerFunc, elemType, f, tileF, tileX, FusedFacs, DistributeToGPUs, RegM, RegK, RegQ, 
                      OptLevel, opX, opF, getKernelFunc, NumThreads, AAlignment, KronAlignment),
                       sm(sm) {
  }
  //TODO: Make "const HardwareDetails"
  virtual bool canCompute(KMMProblem problem, HardwareDetails* hardware, bool p2p, bool exactFuse = true) {
    if (GPUKMMKernel::canCompute(problem, hardware, p2p, exactFuse)) {
      return ((CUDAArchDetails*)hardware)->smArch == sm;
    }
    return false;
  }

  uint32_t ptxVersion() const {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    return attr.ptxVersion;
  }

  uint32_t localSize() const {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    return attr.localSizeBytes;
  }
  uint32_t numRegs()   const {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    return attr.numRegs;
  }

  cudaError_t setSharedMemAttr() {
    cudaError_t err = cudaSuccess;
    if (getMaxSharedMemSize() >= (48 << 10)) {
      err = cudaFuncSetAttribute(kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 getMaxSharedMemSize());
    }

    return err;
  }

  virtual std::string backend() const {
    return "cuda";
  }

  virtual std::string arch() const {
    return smArchToStr(sm);
  }
};