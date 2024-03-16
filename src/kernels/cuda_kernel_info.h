#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu_kernel_info.h"

struct CUDAKernel : public GPUKernel {
  CUDAKernel() {}
  CUDAKernel(void* invokerFunc, Factor f, Factor tileF, Matrix tileX, 
             uint FusedFacs, bool DistributeToGPUs,
             uint RegK, uint RegQ, ElementType elemType,
             fastKronOp opX, fastKronOp opF,
             void*(*getKernelFunc)(), uint NumThreads,
             uint AAlignment, uint KronAlignment) :
             GPUKernel(invokerFunc, f, tileF, tileX, FusedFacs, DistributeToGPUs, RegK, RegQ, 
                       elemType, opX, opF, getKernelFunc, NumThreads, AAlignment, KronAlignment) {}

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