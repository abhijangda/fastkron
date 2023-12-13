#include "handle/handle.h"

#pragma once

class Autotuner {
  FastKronHandle& fastKron;

  cudaError_t tuneSlicedMulSeries(KMMProblem problem,
                                  bool isDistributed, DistributedParams distParams,
                                  std::unordered_map<SlicedMulShape, std::pair<KernelInfo, float>>& bestKernels,
                                  cudaStream_t stream);
public:
  Autotuner(FastKronHandle& fastKron) : fastKron(fastKron)
  {}

  cudaError_t tune(KMMProblem problem, cudaStream_t stream);

};