#include "handle.h"

#pragma once

struct Autotuner {
  cudaError_t tune(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], 
                     uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                     cudaStream_t stream);

  cudaError_t tune(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], 
                     uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                     cudaStream_t stream);

  cudaError_t tune(FastKronHandle& handle, const uint NumKronMats, double* x, double* kronMats[], 
                     uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                     cudaStream_t stream);  
};