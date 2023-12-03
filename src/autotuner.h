#include "handle.h"

#pragma once

class Autotuner {
  FastKronHandle& fastKron;

public:
  Autotuner(FastKronHandle& fastKron) : fastKron(fastKron)
  {}

  cudaError_t tune(const uint NumKronMats, void* x, void** kronMats, 
                       uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                       cudaStream_t stream);

  cudaError_t tuneSlicedMulSeries(KMMProblem problem,
                              void* temp1, void* temp2,
                              bool isDistributed, DistributedParams distParams,
                              std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>>& bestKernels,
                              cudaStream_t stream);
};