#include <cassert>

#include <iostream>
#include <unordered_map>

#include "handle/handle.h"
#include "autotuner/autotuner.h"

/**************************************************
          Library Functions
***************************************************/
cudaError_t fastKronInit(fastKronHandle* handle, fastKronBackend backend, int gpus, int gpusInM, int gpusInK, int gpuLocalKrons) {
  printf("%d\n", backend);
  switch (backend) {
    case fastKronBackend_CUDA:
      printf("15\n");
      #ifndef ENABLE_CUDA
        return cudaErrorInvalidValue;
      #endif
      break;
    case fastKronBackend_ROCM:
      #ifndef ENABLE_ROCM 
        return cudaErrorInvalidValue;
      #endif
      break;
    case fastKronBackend_X86:
      #ifndef ENABLE_X86
        return cudaErrorInvalidValue;
      #endif
      break;
    case fastKronBackend_ARM:
      #ifndef ENABLE_ARM
        return cudaErrorInvalidValue;
      #endif
      break;
    default:
      return cudaErrorInvalidValue;
  }

  FastKronHandle* h = new FastKronHandle(backend, gpus, gpusInM, gpusInK, gpuLocalKrons);
  *handle = h;
  return cudaSuccess;
}

void fastKronDestroy(fastKronHandle handle) {
  handle->free();
  delete handle;
}

cudaError_t gekmmSizes(fastKronHandle handlePtr, uint M, uint N, uint Ps[], uint Qs[], 
                       size_t* resultSize, size_t* tempSize) {
  KMMProblem problem(M, N, Ps, Qs, fastKronOp_N, fastKronOp_N);
  return handlePtr->gekmmSizes(problem, resultSize, tempSize);
}

cudaError_t sgekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], float* X, 
                   fastKronOp opX, float* Fs[], fastKronOp opFs, float* Y,
                   float alpha, float beta, float *Z, float* temp1, float* temp2, cudaStream_t stream) {
  KMMProblem problem(M, N, Ps, Qs, (void*)X, opX, (void**)Fs, opFs, (void*)Y);
  return handle->xgekmm(problem, (void*)temp1, (void*)temp2,
                        EpilogueParams::create<float>(alpha, beta, Z), stream);
}
cudaError_t igekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], int* X, 
                   fastKronOp opX, int* Fs[], fastKronOp opFs, int* Y,
                   int alpha, int beta, int *Z, int* temp1, int* temp2, cudaStream_t stream) {
  KMMProblem problem(M, N, Ps, Qs, (void*)X, opX, (void**)Fs, opFs, (void*)Y);
  return handle->xgekmm(problem, (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<int>(alpha, beta, Z), stream);
}
cudaError_t dgekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], double* X, 
                   fastKronOp opX, double* Fs[], fastKronOp opFs, double* Y,
                   double alpha, double beta, double *Z, double* temp1, double* temp2, cudaStream_t stream) {
  KMMProblem problem(M, N, Ps, Qs, (void*)X, opX, (void**)Fs, opFs, (void*)Y);
  return handle->xgekmm(problem, (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<double>(alpha, beta, Z), stream);
}

cudaError_t sgekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
                       fastKronOp opX, fastKronOp opFs, cudaStream_t stream) {
  return Autotuner(*handle).tune(KMMProblem(M, N, Ps, Qs, opX, opFs), stream);
}
cudaError_t dgekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
                       fastKronOp opX, fastKronOp opFs, cudaStream_t stream) {
  return Autotuner(*handle).tune(KMMProblem(M, N, Ps, Qs, opX, opFs), stream);
}
cudaError_t igekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
                       fastKronOp opX, fastKronOp opFs, cudaStream_t stream) {
  return Autotuner(*handle).tune(KMMProblem(M, N, Ps, Qs, opX, opFs), stream);
}

cudaError_t kronDistributedSGEMM(fastKronHandle handle, const uint NumKronMats, float* x[], float* kronMats[], float* result[],
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
                                 cudaStream_t streams[]) {
  return handle->distributedsgekmm(NumKronMats, x, kronMats, result, M, N, K, 
                                   KronMatCols, KronMatRows, temp1, temp2, streams);
}

cudaError_t allocDistributedX(fastKronHandle handle, float* dX[], float* hX, uint M, uint K) {
  return handle->allocDistributedX((void**)dX, (void*)hX, M, K);
}
cudaError_t gatherDistributedY(fastKronHandle handle, float* dY[], float* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  return handle->gatherDistributedY((void**)dY, (void*)hY, M, K, NumKronMats, KronMatCols, KronMatRows);
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
