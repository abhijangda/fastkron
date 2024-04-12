#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels/gpu_kernel_info.h"

struct CUDAKernel : public GPUKernel {
  SMArch arch;
  CUDAKernel() {}
  CUDAKernel(SMArch arch, void* invokerFunc, Factor f, Factor tileF, Matrix tileX, 
             uint FusedFacs, bool DistributeToGPUs,
             uint RegM, uint RegK, uint RegQ, FastKronType elemType, uint OptLevel,
             fastKronOp opX, fastKronOp opF,
             void*(*getKernelFunc)(), uint NumThreads,
             uint AAlignment, uint KronAlignment) :
             GPUKernel(invokerFunc, f, tileF, tileX, FusedFacs, DistributeToGPUs, RegM, RegK, RegQ, 
                       elemType, OptLevel, opX, opF, getKernelFunc, NumThreads, AAlignment, KronAlignment),
                       arch(arch) {
  }
  //TODO: Make "const HardwareDetails"
  virtual bool canCompute(KMMProblem problem, HardwareDetails* hardware, bool p2p, bool exactFuse = true) {
    if (GPUKernel::canCompute(problem, hardware, p2p, exactFuse)) {
      return ((CUDAArchDetails*)hardware)->smArch == arch;
    }
    return false;
  }

  uint32_t ptxVersion() const {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernelFunc));
    return attr.ptxVersion;
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

  virtual std::string runtimeStr() const {
    return "cuda";
  }

  virtual std::string archStr() const {
    return smArchToStr(arch);
  }
};