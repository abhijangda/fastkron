#include <cassert>

#include <iostream>
#include <unordered_map>

#include "handle.h"
#include "device/params.h"
#include "env.h"
#include "device/kernel_info.h"
#include "autotuner.h"
#include "utils.h"
#include "kmmalgo.h"

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

cudaError_t sgekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], float* X, float* Fs[], float* Y,
                    float alpha, float beta, float *Z, float* temp1, float* temp2, cudaStream_t stream) {
  return handle->xgekmm(M, N, Ps, Qs, (void*)X, (void**)Fs, (void*)Y,
                        (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<float>(alpha, beta, Z), stream);
}
cudaError_t igekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], int* X, int* Fs[], int* Y,
                   int alpha, int beta, int *Z, int* temp1, int* temp2, cudaStream_t stream) {
  return handle->xgekmm(M, N, Ps, Qs, (void*)X, (void**)Fs, (void*)Y,
                        (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<int>(alpha, beta, Z), stream);
}
cudaError_t dgekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], double* X, double* Fs[], double* Y,
                   double alpha, double beta, double *Z, double* temp1, double* temp2, cudaStream_t stream) {
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

cudaError_t sgekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], cudaStream_t stream) {
  return Autotuner(*handle).tune(M, N, Ps, Qs, stream);
}
cudaError_t dgekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], cudaStream_t stream) {
  return Autotuner(*handle).tune(M, N, Ps, Qs, stream);
}
cudaError_t igekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], cudaStream_t stream) {
  return Autotuner(*handle).tune(M, N, Ps, Qs, stream);
}


cudaError_t allocDistributedX(fastKronHandle handle, float* dX[], float* hX, uint M, uint K) {
  return handle->allocDistributedX((void**)dX, (void*)hX, M, K);
}
cudaError_t gatherDistributedY(fastKronHandle handle, float* dY[], float* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  return handle->gatherDistributedY((void**)dY, (void*)hY, M, K, NumKronMats, KronMatCols, KronMatRows);
}

cudaError_t gekmmSizes(fastKronHandle handlePtr, uint M, uint N, uint Ps[], uint Qs[], 
                       size_t* resultSize, size_t* tempSize) {
  if (resultSize == nullptr) return cudaErrorInvalidValue;
  if (tempSize   == nullptr) return cudaErrorInvalidValue;

  uint gpuM, gpuK;

  const uint K = std::reduce(Ps, Ps + N, 1, std::multiplies<uint>());
  const uint L = std::reduce(Qs, Qs + N, 1, std::multiplies<uint>());

  FastKronHandle& handle = *handlePtr;
  KMMProblem problem(M, N, Ps, Qs);
  if (handle.isDistributed_) {
    if (!checkDistributedKronSizes(problem, handle.perGPUKronBatch_, handle.gpusInK_))
      return cudaErrorInvalidValue;
    gpuM = M/handle.gpusInM_;
    gpuK = K/handle.gpusInK_;
  } else {
    gpuM = M;
    gpuK = K;
  }

  int maxTempN = 0;
  int resultCols = 0;
                     
  auto e = executeGeKMM(problem, nullptr, nullptr,
    [](const KMMProblem kmm) {return 1;},
    [&maxTempN, &resultCols](const KMMProblem kmm, int rstart, void* temps[2], void* result) {
                            maxTempN = std::max(maxTempN, std::max(kmm.k, kmm.l));
                            resultCols = kmm.l;
                            return cudaSuccess;
                          });
  *tempSize   = gpuM * maxTempN;
  if (handle.isDistributed_ and handle.distComm_ == DistComm::NCCL)
    //Include size of send and recv buffers 
    *tempSize = (*tempSize) * 2;
  *resultSize = gpuM * resultCols;

  return e;
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
