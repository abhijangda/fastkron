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
  return handle->sgekmm(NumKronMats, x, kronMats, result,
                                            M, N, K, KronMatCols, KronMatRows, temp1, temp2, 
                                            EpilogueParams<float>(alpha, beta, z), stream);
}

cudaError_t igekmm(fastKronHandle handle, const uint NumKronMats, int* x, int* kronMats[], int* result,
                   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], int* temp1, int* temp2,
                   int alpha, int beta, int *z, cudaStream_t stream) {
  return handle->igekmm(NumKronMats, x, kronMats, result, 
                                        M, N, K, KronMatCols, KronMatRows, temp1, temp2,
                                        EpilogueParams<int>(alpha, beta, z), stream);
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

cudaError_t gekmmSizes(fastKronHandle handlePtr, const uint NumKronMats, uint M, uint N, uint K, 
                          uint KronMatCols[], uint KronMatRows[], size_t* resultSize, size_t* tempSize) {
  if (resultSize == nullptr) return cudaErrorInvalidValue;
  if (tempSize   == nullptr) return cudaErrorInvalidValue;
  uint gpuM, gpuK;
  FastKronHandle& handle = *handlePtr;
  if (handle.isDistributed_) {
    if (!checkDistributedKronSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows, 
                                   handle.perGPUKronBatch_, handle.gpusInK_))
      return cudaErrorInvalidValue;
    gpuM = M/handle.gpusInM_;
    gpuK = K/handle.gpusInK_;
  } else {
    if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
      return cudaErrorInvalidValue;
    gpuM = M;
    gpuK = K;
  }
  size_t tempN = gpuK;
  size_t maxTempN = tempN;
  for (int i = NumKronMats - 1; i >= 0; i--) {
    tempN = (tempN/KronMatRows[i])*KronMatCols[i];
    if (maxTempN < tempN)
      maxTempN = tempN;
  }

  *tempSize   = gpuM * maxTempN;
  if (handle.isDistributed_ and handle.distComm_ == DistComm::NCCL)
    //Include size of send and recv buffers 
    *tempSize = (*tempSize) * 2;
  *resultSize = gpuM * tempN;

  return cudaSuccess;
}

cudaError_t sgekmmTune(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], 
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                                 cudaStream_t stream) {
  return Autotuner().tune(*handle, NumKronMats, x, kronMats,
                         M, N, K, KronMatCols, KronMatRows,
                         stream);
}

cudaError_t dgekmmTune(fastKronHandle handle, const uint NumKronMats, double* x, double* kronMats[], 
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream) {
  return Autotuner().tune(*handle, NumKronMats, x, kronMats, 
                          M, N, K, KronMatCols, KronMatRows,
                          stream);
}

cudaError_t idgemmTune(fastKronHandle handle, const uint NumKronMats, int* x, int* kronMats[],
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream) {
  return Autotuner().tune(*handle, NumKronMats, x, kronMats,
                       M, N, K, KronMatCols, KronMatRows,
                       stream);
}

void FastKronHandle::getDistributedSizes(uint M, uint K, uint& gpuM, uint& gpuK) {
  gpuM = M/gpusInM_;
  gpuK = K/gpusInK_;
}