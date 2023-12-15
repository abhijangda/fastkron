#include "kmm/kmmalgo.h"
#include "device/kernel_info.h"
#include "device/params.h"

#pragma once

class KernelInvoker {
private:
    std::unordered_map<Factor, std::vector<KernelInfo>> compiledKernels;

public:
  cudaError_t invokeKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                           KMMProblem problem,
                           EpilogueParams epilogueParams,
                           cudaStream_t stream);
  cudaError_t invokeP2PStoreKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                   KMMProblem problem, DistributedParams distParams, 
                                   EpilogueParams epilogueParams,
                                   cudaStream_t stream);
};