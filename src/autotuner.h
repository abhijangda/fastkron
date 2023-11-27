#include "handle.h"

#pragma once

class Autotuner {
  FastKronHandle& fastKron;

public:
  Autotuner(FastKronHandle& fastKron) : fastKron(fastKron)
  {}

  cudaError_t tune(FastKronHandle& handle, const uint NumKronMats, void* x, void** kronMats, 
                       uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                       cudaStream_t stream);
};