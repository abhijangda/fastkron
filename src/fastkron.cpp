#include <cassert>

#include <iostream>
#include <unordered_map>

#include "handle/handle.h"
#include "autotuner/autotuner.h"

/**************************************************
          Library Functions
***************************************************/
fastKronError fastKronInit(fastKronHandle* handle, uint32_t backends) {
  FastKronHandle* h = new FastKronHandle(backends);
  *handle = (fastKronHandle)h;
  uint32_t fastKronOptionsAll = fastKronOptionsUseFusion;
  fastKronSetOptions(*handle, fastKronOptionsAll);

  return (h->hasBackend(fastKronBackend_X86) || 
          h->hasBackend(fastKronBackend_CUDA) ||
          h->hasBackend(fastKronBackend_HIP)) ? 
          fastKronSuccess : fastKronInvalidArgument;
}

fastKronError fastKronSetOptions(fastKronHandle handle, uint32_t options) {
  ((FastKronHandle*)handle)->setOptions(options);
  return fastKronSuccess;
}

void fastKronDestroy(fastKronHandle handle) {
  ((FastKronHandle*)handle)->free();
  delete (FastKronHandle*)handle;
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
  return ((FastKronHandle*)handlePtr)->initCUDABackend(ptrToStream, gpus, gpusInM, gpusInK, gpuLocalKrons);
}

fastKronError fastKronInitHIP(fastKronHandle handlePtr, void *ptrToStream) {
  //TODO: remove Backend
  return ((FastKronHandle*)handlePtr)->initHIPBackend(ptrToStream);
}

fastKronError fastKronInitX86(fastKronHandle handlePtr) {
  return ((FastKronHandle*)handlePtr)->initX86Backend();
}

fastKronError gekmmSizes(fastKronHandle handlePtr, uint M, uint N, uint Ps[], uint Qs[], 
                       size_t* resultSize, size_t* tempSize) {
  KMMProblem problem(FastKronTypeNone, M, N, Ps, Qs, fastKronOp_N, fastKronOp_N);
  return ((FastKronHandle*)handlePtr)->gekmmSizes(problem, resultSize, tempSize);
}

fastKronError sgekmm(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], float* X, 
                   fastKronOp opX, float* Fs[], fastKronOp opFs, float* Y,
                   float alpha, float beta, float *Z, float* temp1, float* temp2) {
  KMMProblem problem(FastKronFloat, M, N, Ps, Qs, (void*)X, opX, (void**)Fs, opFs, (void*)Y);
  return ((FastKronHandle*)handle)->xgekmm(problem, backend, (void*)temp1, (void*)temp2,
                        EpilogueParams::create<float>(alpha, beta, Z));
}
fastKronError igekmm(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], int* X, 
                   fastKronOp opX, int* Fs[], fastKronOp opFs, int* Y,
                   int alpha, int beta, int *Z, int* temp1, int* temp2) {
  KMMProblem problem(FastKronInt, M, N, Ps, Qs, (void*)X, opX, (void**)Fs, opFs, (void*)Y);
  return ((FastKronHandle*)handle)->xgekmm(problem, backend, (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<int>(alpha, beta, Z));
}
fastKronError dgekmm(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], double* X, 
                   fastKronOp opX, double* Fs[], fastKronOp opFs, double* Y,
                   double alpha, double beta, double *Z, double* temp1, double* temp2) {
  KMMProblem problem(FastKronDouble, M, N, Ps, Qs, (void*)X, opX, (void**)Fs, opFs, (void*)Y);
  return ((FastKronHandle*)handle)->xgekmm(problem, backend, (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<double>(alpha, beta, Z));
}

// fastKronError sgekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
//                        fastKronOp opX, fastKronOp opFs) {
//   return Autotuner(*handle).tune(KMMProblem(FastKronFloat, M, N, Ps, Qs, opX, opFs));
// }
// fastKronError dgekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
//                        fastKronOp opX, fastKronOp opFs) {
//   return Autotuner(*handle).tune(KMMProblem(FastKronDouble, M, N, Ps, Qs, opX, opFs));
// }
// fastKronError igekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
//                        fastKronOp opX, fastKronOp opFs) {
//   return Autotuner(*handle).tune(KMMProblem(FastKronInt, M, N, Ps, Qs, opX, opFs));
// }

fastKronError kronDistributedSGEMM(fastKronHandle handle, const uint NumKronMats, void* x[], void* kronMats[], void* result[],
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], void** temp1, void** temp2,
                                 void* streams) {
  return ((FastKronHandle*)handle)->distributedsgekmm(NumKronMats, (float**)x, (float**)kronMats, (float**)result, M, N, K, 
                                   KronMatCols, KronMatRows, (float**)temp1, (float**)temp2, streams);
}

fastKronError allocDistributedX(fastKronHandle handle, void* dX[], void* hX, uint M, uint K) {
  return ((FastKronHandle*)handle)->allocDistributedX((void**)dX, (void*)hX, M, K);
}
fastKronError gatherDistributedY(fastKronHandle handle, void* dY[], void* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  return ((FastKronHandle*)handle)->gatherDistributedY((void**)dY, (void*)hY, M, K, NumKronMats, KronMatCols, KronMatRows);
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
