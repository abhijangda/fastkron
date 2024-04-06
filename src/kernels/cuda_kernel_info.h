#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels/gpu_kernel_info.h"

struct CUDAKernel : public GPUKernel {
  CUDAKernel() {}
  CUDAKernel(void* invokerFunc, Factor f, Factor tileF, Matrix tileX, 
             uint FusedFacs, bool DistributeToGPUs,
             uint RegM, uint RegK, uint RegQ, ElementType elemType, uint OptLevel,
             fastKronOp opX, fastKronOp opF,
             void*(*getKernelFunc)(), uint NumThreads,
             uint AAlignment, uint KronAlignment) :
             GPUKernel(invokerFunc, f, tileF, tileX, FusedFacs, DistributeToGPUs, RegM, RegK, RegQ, 
                       elemType, OptLevel, opX, opF, getKernelFunc, NumThreads, AAlignment, KronAlignment) {
  }
  uint32_t localSize() const {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernelFunc));
    return attr.localSizeBytes;
  }
  uint32_t numRegs()   const {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernelFunc));
    return attr.numRegs;
  }

  cudaError_t setSharedMemAttr() {
    cudaError_t err = cudaSuccess;
    if (sharedMemSize() >= (48 << 10)) {
      err = cudaFuncSetAttribute(kernelFunc,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 sharedMemSize());
    }

    return err;
  }
};