#include <cassert>

#include <iostream>
#include <unordered_map>

#include "handle/handle.h"
#include "autotuner/autotuner.h"

/**************************************************
          Library Functions
***************************************************/
fastKronError fastKronInit(fastKronHandle* handle, fastKronBackend backend) {
  switch (backend) {
    case fastKronBackend_CUDA:
      #ifndef ENABLE_CUDA
        return fastKronBackendNotAvailable;
      #endif
      break;
    case fastKronBackend_HIP:
      #ifndef ENABLE_ROCM 
        return fastKronBackendNotAvailable;
      #endif
      break;
    case fastKronBackend_X86:
      #ifndef ENABLE_X86
        return fastKronBackendNotAvailable;
      #endif
      break;
    case fastKronBackend_ARM:
      #ifndef ENABLE_ARM
        return fastKronBackendNotAvailable;
      #endif
      break;
    default:
      return fastKronBackendNotAvailable;
  }

  FastKronHandle* h = new FastKronHandle(backend);
  *handle = h;
  return fastKronSuccess;
}

void fastKronDestroy(fastKronHandle handle) {
  handle->free();
  delete handle;
}

const char* fastKronGetErrorString(fastKronError err) {
  switch(err) {
    case fastKronSuccess:
      return "fastKronSuccess";
    case fastKronBackendNotAvailable:
      return "fastKronBackendNotAvailable";
    case fastKronInvalidMemoryAccess:
      return "fastKronInvalidMemoryAccess";
    case fastKronKernelNotFound:
      return "fastKronKernelNotFound";
    case fastKronInvalidArgument:
      return "fastKronInvalidArgument";
    case fastKronOtherError:
      return "fastKronOtherError";
    default:
      return NULL;
  }
}

fastKronError fastKronInitCUDA(fastKronHandle handlePtr, void *ptrToStream, int gpus, int gpusInM, int gpusInK, int gpuLocalKrons) {
  return handlePtr->initCUDABackend(ptrToStream, gpus, gpusInM, gpusInK, gpuLocalKrons);
}

fastKronError fastKronInitX86(fastKronHandle handlePtr) {
  return handlePtr->initX86Backend();
}

fastKronError gekmmSizes(fastKronHandle handlePtr, uint M, uint N, uint Ps[], uint Qs[], 
                       size_t* resultSize, size_t* tempSize) {
  KMMProblem problem(M, N, Ps, Qs, fastKronOp_N, fastKronOp_N);
  return handlePtr->gekmmSizes(problem, resultSize, tempSize);
}

fastKronError sgekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], float* X, 
                   fastKronOp opX, float* Fs[], fastKronOp opFs, float* Y,
                   float alpha, float beta, float *Z, float* temp1, float* temp2) {
  KMMProblem problem(M, N, Ps, Qs, (void*)X, opX, (void**)Fs, opFs, (void*)Y);
  return handle->xgekmm(problem, (void*)temp1, (void*)temp2,
                        EpilogueParams::create<float>(alpha, beta, Z));
}
fastKronError igekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], int* X, 
                   fastKronOp opX, int* Fs[], fastKronOp opFs, int* Y,
                   int alpha, int beta, int *Z, int* temp1, int* temp2) {
  KMMProblem problem(M, N, Ps, Qs, (void*)X, opX, (void**)Fs, opFs, (void*)Y);
  return handle->xgekmm(problem, (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<int>(alpha, beta, Z));
}
fastKronError dgekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], double* X, 
                   fastKronOp opX, double* Fs[], fastKronOp opFs, double* Y,
                   double alpha, double beta, double *Z, double* temp1, double* temp2) {
  KMMProblem problem(M, N, Ps, Qs, (void*)X, opX, (void**)Fs, opFs, (void*)Y);
  return handle->xgekmm(problem, (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<double>(alpha, beta, Z));
}

fastKronError sgekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
                       fastKronOp opX, fastKronOp opFs) {
  return Autotuner(*handle).tune(KMMProblem(M, N, Ps, Qs, opX, opFs));
}
fastKronError dgekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
                       fastKronOp opX, fastKronOp opFs) {
  return Autotuner(*handle).tune(KMMProblem(M, N, Ps, Qs, opX, opFs));
}
fastKronError igekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
                       fastKronOp opX, fastKronOp opFs) {
  return Autotuner(*handle).tune(KMMProblem(M, N, Ps, Qs, opX, opFs));
}

fastKronError kronDistributedSGEMM(fastKronHandle handle, const uint NumKronMats, float* x[], float* kronMats[], float* result[],
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
                                 void* streams) {
  return handle->distributedsgekmm(NumKronMats, x, kronMats, result, M, N, K, 
                                   KronMatCols, KronMatRows, temp1, temp2, streams);
}

fastKronError allocDistributedX(fastKronHandle handle, float* dX[], float* hX, uint M, uint K) {
  return handle->allocDistributedX((void**)dX, (void*)hX, M, K);
}
fastKronError gatherDistributedY(fastKronHandle handle, float* dY[], float* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  return handle->gatherDistributedY((void**)dY, (void*)hY, M, K, NumKronMats, KronMatCols, KronMatRows);
}

// fastKronError allocDistributedX(fastKronHandle handle, int* dX[], int* hX, uint M, uint K) {
//   assert(false); handle->allocDistributedX((void**)dX, (void*)hX, M, K);
// }
// fastKronError gatherDistributedY(fastKronHandle handle, int* dY[], int* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
//   assert(false); handle->gatherDistributedY((void**)dY, (void*)hY, M, K, NumKronMats, KronMatCols, KronMatRows);
// }

// fastKronError allocDistributedX(fastKronHandle handle, double* dX[], double* hX, uint M, uint K) {
//   assert(false);handle->allocDistributedX((void**)dX, (void*)hX, M, K);
// }
// fastKronError gatherDistributedY(fastKronHandle handle, double* dY[], double* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
//   assert(false);handle->gatherDistributedY((void**)dY, (void*)hY, M, K, NumKronMats, KronMatCols, KronMatRows);
// }
