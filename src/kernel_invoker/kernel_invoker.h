#include "device/kernel_info.h"
#include "device/params.h"

#pragma once

class KernelInvoker {
public:
  cudaError_t fusedSlicedMatmul(uint NumFusedKerns, KernelInfo& kernelInfo, const uint kronIndex, 
                              void* x, void** krons, void* kronGemmResult,
                              const uint M, const uint N, const uint K, 
                              const uint* FusedKronMatCols, const uint* FusedKronMatRows,
                              EpilogueParams epilogueParams,
                              cudaStream_t stream);
  cudaError_t fusedDistributedSlicedMatmul(const uint NumFusedKerns, KernelInfo& kernel, const uint kronIndex, 
                                           void* x, void** kronMat, void* kronGemmResult,
                                           const uint M, const uint N, const uint K, 
                                           const uint* FusedKronMatCols, const uint* FusedKronMatRows,
                                           DistributedParams distParams, EpilogueParams epilogueParams,
                                           cudaStream_t stream);
};