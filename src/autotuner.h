#include "handle.h"

#pragma once

class Autotuner {
  FastKronHandle& fastKron;

public:
  Autotuner(FastKronHandle& fastKron) : fastKron(fastKron)
  {}

  cudaError_t tune(uint M, uint N, uint Ps[], uint Qs[], cudaStream_t stream);

  cudaError_t tuneSlicedMulSeries(KMMProblem problem,
                                  bool isDistributed, DistributedParams distParams,
                                  std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>>& bestKernels,
                                  cudaStream_t stream);
};