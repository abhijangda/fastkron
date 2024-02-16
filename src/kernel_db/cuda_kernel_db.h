#include <functional>

#include <cuda_runtime.h>
#include <cuda.h>

#include "kmm/kmmalgo.h"
#include "kernels/kernel_info.h"
#include "kernels/params.h"
#include "kernel_db/kernel_db.h"

#pragma once

//TODO: Change name to Executor?
class CUDAKernelDatabase : public KernelDatabase {
  cudaStream_t stream;

public:
  CUDAKernelDatabase();

  void setStream(cudaStream_t stream) {
    this->stream = stream;
  }

  void free() {
    compiledKernels.clear();
  }
  
  virtual cudaError_t invokeKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                   KMMProblem problem,
                                   EpilogueParams epilogueParams,
                                   KernelMode execMode);
  virtual cudaError_t invokeP2PStoreKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                           KMMProblem problem, DistributedParams distParams, 
                                           EpilogueParams epilogueParams,
                                           KernelMode execMode, cudaStream_t stream); //TODO: Fix remove stream arg

  std::pair<KernelInfo, float> tuneKernelForProblem(KMMProblem problem, bool distP2PStore, uint factorIdx, DistributedParams distParams);
  virtual cudaError_t procMalloc(uint32_t proc, size_t size, void*& ptr);
  virtual cudaError_t procMalloc(uint32_t proc, Matrix& m);
  virtual cudaError_t procMemset(uint32_t proc, Matrix& m, float val);
  virtual cudaError_t procFree(uint32_t proc, Matrix m);
  virtual cudaError_t procFree(uint32_t proc, void* ptr);
};