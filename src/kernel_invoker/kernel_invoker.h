#include "device/kernel_info.h"
#include "device/params.h"

#pragma once

class KernelInvoker {
public:
  cudaError_t fusedSlicedMatmul(KernelInfo& kernelInfo, const uint kronIndex, 
                                KMMProblem problem,
                                EpilogueParams epilogueParams,
                                cudaStream_t stream);
  cudaError_t fusedDistributedSlicedMatmul(KernelInfo& kernelInfo, const uint kronIndex, 
                                           KMMProblem problem, DistributedParams distParams, 
                                           EpilogueParams epilogueParams,
                                           cudaStream_t stream);
};