#include <cassert>

#include <iostream>
#include <unordered_map>

#include "handle/handle.h"
#include "autotuner/autotuner.h"

#include "fastkron.h"

#ifdef ENABLE_MULTI_GPU
  #include "fastkronMg.h"
#endif

/**************************************************
          Library Functions
***************************************************/
fastKronError fastKronInit(fastKronHandle* handle, uint32_t backends) {
  FastKronHandle* h = new FastKronHandle(backends);
  *handle = (fastKronHandle)h;
  uint32_t fastKronOptionsAll = fastKronOptionsUseFusion;
  fastKronSetOptions(*handle, fastKronOptionsAll);

  if (backends & fastKronGetBackends() != backends)
    return fastKronInvalidArgument;

  return fastKronSuccess;
}

fastKronError fastKronInitAllBackends(fastKronHandle* handle) {
  return fastKronInit(handle, fastKronGetBackends());
}

fastKronError fastKronSetOptions(fastKronHandle handle, uint32_t options) {
  ((FastKronHandle*)handle)->setOptions(options);
  return fastKronSuccess;
}

void fastKronDestroy(fastKronHandle handle) {
  ((FastKronHandle*)handle)->free();
  delete (FastKronHandle*)handle;
}

uint32_t fastKronGetBackends() {
  uint32_t backends = 0;
  #ifdef ENABLE_CUDA
    backends |= fastKronBackend_CUDA;
  #endif
  #ifdef ENABLE_X86
    backends |= fastKronBackend_X86;
  #endif
  #ifdef ENABLE_HIP
    backends |= fastKronBackend_HIP;
  #endif

  return backends;
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

fastKronError fastKronInitCUDA(fastKronHandle handlePtr, void *ptrToStream) {
  return ((FastKronHandle*)handlePtr)->initCUDABackend(ptrToStream, 1, 1, 1, 1);
}

fastKronError fastKronMgInitCUDA(fastKronHandle handlePtr, void *streams, int gpus, int gpusInM, int gpusInK, int gpuLocalKrons) {
  return ((FastKronHandle*)handlePtr)->initCUDABackend(streams, gpus, gpusInM, gpusInK, gpuLocalKrons);
}

fastKronError fastKronInitHIP(fastKronHandle handlePtr, void *ptrToStream) {
  //TODO: remove Backend
  return ((FastKronHandle*)handlePtr)->initHIPBackend(ptrToStream);
}

fastKronError fastKronInitX86(fastKronHandle handlePtr) {
  return ((FastKronHandle*)handlePtr)->initX86Backend();
}

fastKronError fastKronSetStream(fastKronHandle handlePtr, fastKronBackend backend,
                                void* ptrToStream) {
  return ((FastKronHandle*)handlePtr)->setStream(backend, ptrToStream);
}

fastKronError gekmmSizes(fastKronHandle handlePtr, uint M, uint N, uint Ps[], uint Qs[], 
                       size_t* resultSize, size_t* tempSize) {
  KMMProblem problem(FastKronTypeNone, M, N, Ps, Qs, fastKronOp_N, fastKronOp_N);
  return ((FastKronHandle*)handlePtr)->gekmmSizes(problem, resultSize, tempSize);
}

fastKronError sgekmm(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], const float* X,
                   fastKronOp opX, const float* Fs[], fastKronOp opFs, float* Y,
                   float alpha, float beta, const float *Z, float* temp1, float* temp2) {
  KMMProblem problem(FastKronFloat, M, N, Ps, Qs, (void*)X, opX, (void**)Fs, opFs, (void*)Y);
  return ((FastKronHandle*)handle)->xgekmm(problem, backend, (void*)temp1, (void*)temp2,
                        EpilogueParams::create<float>(alpha, beta, Z));
}
fastKronError igekmm(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], const int* X,
                   fastKronOp opX, const int* Fs[], fastKronOp opFs, int* Y,
                   int alpha, int beta, const int *Z, int* temp1, int* temp2) {
  KMMProblem problem(FastKronInt, M, N, Ps, Qs, (void*)X, opX, (void**)Fs, opFs, (void*)Y);
  return ((FastKronHandle*)handle)->xgekmm(problem, backend, (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<int>(alpha, beta, Z));
}
fastKronError dgekmm(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], const double* X,
                   fastKronOp opX, const double* Fs[], fastKronOp opFs, double* Y,
                   double alpha, double beta, const double *Z, double* temp1, double* temp2) {
  KMMProblem problem(FastKronDouble, M, N, Ps, Qs, (void*)X, opX, (void**)Fs, opFs, (void*)Y);
  return ((FastKronHandle*)handle)->xgekmm(problem, backend, (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<double>(alpha, beta, Z));
}

#ifdef ENABLE_MULTI_GPU
fastKronError fastKronMgSGEMM(fastKronHandle handle, const uint NumKronMats, void* x[], void* kronMats[], void* result[],
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], void** temp1, void** temp2,
                                 void* streams) {
  return ((FastKronHandle*)handle)->distributedsgekmm(NumKronMats, (float**)x, (float**)kronMats, (float**)result, M, N, K, 
                                   KronMatCols, KronMatRows, (float**)temp1, (float**)temp2, streams);
}

fastKronError fastKronMgAllocX(fastKronHandle handle, void* dX[], void* hX, uint M, uint K) {
  return ((FastKronHandle*)handle)->allocDistributedX((void**)dX, (void*)hX, M, K);
}

fastKronError fastKronMgGatherY(fastKronHandle handle, void* dY[], void* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  return ((FastKronHandle*)handle)->gatherDistributedY((void**)dY, (void*)hY, M, K, NumKronMats, KronMatCols, KronMatRows);
}
#endif