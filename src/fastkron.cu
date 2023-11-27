#include <cassert>

#include <iostream>
#include <unordered_map>

#include "handle.h"
#include "device/params.h"
#include "env.h"
#include "device/kernel_info.h"
#include "autotuner.h"
#include "utils.h"

/**************************************************
          Library Functions
***************************************************/
cudaError_t fastKronInit(fastKronHandle* handle, int gpus, int gpusInM, int gpusInK, int gpuLocalKrons) {
  FastKronHandle* h = new FastKronHandle(gpus, gpusInM, gpusInK, gpuLocalKrons);
  *handle = h;
  return cudaSuccess;
}

void fastKronDestroy(fastKronHandle handle) {
  handle->free();
  delete handle;
}

cudaError_t sgekmm(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], float* result,
                   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float* temp1, float* temp2,
                   float alpha, float beta, float *z, cudaStream_t stream) {
  return handle->xgekmm(NumKronMats, (void*)x, (void**)kronMats, (void*)result,
                        M, N, K, KronMatCols, KronMatRows, (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<float>(alpha, beta, z), stream);
}

cudaError_t igekmm(fastKronHandle handle, const uint NumKronMats, int* x, int* kronMats[], int* result,
                   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], int* temp1, int* temp2,
                   int alpha, int beta, int *z, cudaStream_t stream) {
  return handle->xgekmm(NumKronMats, (void*)x, (void**)kronMats, (void*)result, 
                        M, N, K, KronMatCols, KronMatRows, (void*)temp1, (void*)temp2,
                        EpilogueParams::create<int>(alpha, beta, z), stream);
}

cudaError_t dgekmm(fastKronHandle handle, const uint NumKronMats, double* x, double* kronMats[], double* result,
                   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], double* temp1, double* temp2,
                   double alpha, double beta, double *z, cudaStream_t stream) {
  return cudaSuccess;
                    // return handle->gekmm(FastKronType::Double, NumKronMats, x, kronMats, result, 
  //                                             M, N, K, KronMatCols, KronMatRows, temp1, temp2,
  //                                             EpilogueParams<double>(alpha, beta, z), stream);
}


cudaError_t kronSGEMMOutofCore(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream) {
  // return singleGPUOutOfCoreKronMatmul<float, float4>(handle, NumKronMats, x, kronMats, result, 
  //                                                    M, N, K, KronMatCols, KronMatRows, stream);
}

// cudaError_t kronSGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
//   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]) {
//   return singleGPUOutOfCoreKronMatmul<float, float4>(handle, NumKronMats, x, kronMats, result, 
//                                                      M, N, K, KronMatCols, KronMatRows, stream);
// }

// cudaError_t kronIGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], int** result,
//   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]) {
//   return singleGPUOutOfCoreKronMatmul<int, int4>(handle, NumKronMats, x, kronMats, result, 
//                                                  M, N, K, KronMatCols, KronMatRows, stream);
// }

cudaError_t kronDistributedSGEMM(fastKronHandle handle, const uint NumKronMats, float* x[], float* kronMats[], float* result[],
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
                                 cudaStream_t streams[]) {
  return handle->distributedsgekmm(NumKronMats, x, kronMats, result, M, N, K, 
                                   KronMatCols, KronMatRows, temp1, temp2, streams);
}

cudaError_t sgekmmTune(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], 
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                                 cudaStream_t stream) {
  return Autotuner(*handle).tune(NumKronMats, (void*)x, (void**)kronMats,
                                 M, N, K, KronMatCols, KronMatRows,
                                 stream);
}

cudaError_t dgekmmTune(fastKronHandle handle, const uint NumKronMats, double* x, double* kronMats[], 
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream) {
  return Autotuner(*handle).tune(NumKronMats, (void*)x, (void**)kronMats,
                                 M, N, K, KronMatCols, KronMatRows,
                                 stream);
}

cudaError_t idgemmTune(fastKronHandle handle, const uint NumKronMats, int* x, int* kronMats[],
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream) {
  return Autotuner(*handle).tune(NumKronMats, (void*)x, (void**)kronMats,
                       M, N, K, KronMatCols, KronMatRows,
                       stream);
}


cudaError_t allocDistributedX(fastKronHandle handle, float* dX[], float* hX, uint M, uint K) {
  handle->allocDistributedX((void**)dX, (void*)hX, M, K);
}
cudaError_t gatherDistributedY(fastKronHandle handle, float* dY[], float* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  handle->gatherDistributedY((void**)dY, (void*)hY, M, K, NumKronMats, KronMatCols, KronMatRows);
}

// cudaError_t allocDistributedX(fastKronHandle handle, int* dX[], int* hX, uint M, uint K) {
//   assert(false); handle->allocDistributedX((void**)dX, (void*)hX, M, K);
// }
// cudaError_t gatherDistributedY(fastKronHandle handle, int* dY[], int* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
//   assert(false); handle->gatherDistributedY((void**)dY, (void*)hY, M, K, NumKronMats, KronMatCols, KronMatRows);
// }

// cudaError_t allocDistributedX(fastKronHandle handle, double* dX[], double* hX, uint M, uint K) {
//   assert(false);handle->allocDistributedX((void**)dX, (void*)hX, M, K);
// }
// cudaError_t gatherDistributedY(fastKronHandle handle, double* dY[], double* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
//   assert(false);handle->gatherDistributedY((void**)dY, (void*)hY, M, K, NumKronMats, KronMatCols, KronMatRows);
// }
