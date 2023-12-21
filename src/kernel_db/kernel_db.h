#include <cuda_runtime.h>
#include <cuda.h>

#include "kmm/kmmalgo.h"
#include "device/kernel_info.h"
#include "device/params.h"

#pragma once

class KernelDatabase {
private:
    std::unordered_map<Factor, std::vector<KernelInfo>> compiledKernels;

public:
  KernelDatabase();
  void free() {
    compiledKernels.clear();
  }
  
  cudaError_t invokeKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                           KMMProblem problem,
                           EpilogueParams epilogueParams,
                           cudaStream_t stream);
  cudaError_t invokeP2PStoreKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                   KMMProblem problem, DistributedParams distParams, 
                                   EpilogueParams epilogueParams,
                                   cudaStream_t stream);
  std::pair<KernelInfo, float> tuneKernelForSize(KMMProblem problem, bool distP2PStore, uint factorIdx, DistributedParams distParams, cudaStream_t stream);
};