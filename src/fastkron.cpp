#include <cassert>

#include <iostream>
#include <unordered_map>

#include "handle/handle.h"
#include "autotuner/autotuner.h"

#include "fastkron.h"

#ifdef ENABLE_MULTI_GPU
  #include "fastkronMg.h"
#endif

#define STR_(x) #x
#define STR(x) STR_(x)

const char* fastKronVersion() {
  return 
  ""
  STR(FASTKRON_VERSION) 
#ifdef ENABLE_X86
  "+x86_64"
#endif
#ifdef ENABLE_CUDA
  "+CUDA"
  STR(FASTKRON_CUDA_VERSION)
#endif
  ;
}

const char* fastKronCUDAArchs() {
//   return 
// #ifdef ENABLE_CUDA
//   ""
//   STR(FASTKRON_CUDA_ARCHS);
// #else
//   NULL;
// #endif
return NULL;
}

const char* fastKronGetErrorString(fastKronError err) {
  switch(err) {
    case fastKronSuccess:
      return "Operation successfull";
    case fastKronBackendNotAvailable:
      return "Requested backend is not available";
    case fastKronInvalidMemoryAccess:
      return "Illegal memory access";
    case fastKronKernelNotFound:
      return "Kernel to execute the problem not found";
    case fastKronInvalidArgument:
      return "An argument to the function is invalid";
    case fastKronOtherError:
      return "Unknown error occurred";
    default:
      return NULL;
  }
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

fastKronError fastKronInit(fastKronHandle* handle, uint32_t backends) {
  FastKronHandle* h = new FastKronHandle(backends);
  *handle = (fastKronHandle)h;
  uint32_t fastKronOptionsAll = fastKronOptionsUseFusion;
  fastKronSetOptions(*handle, fastKronOptionsAll);

  if ((backends & fastKronGetBackends()) != backends)
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
  delete (FastKronHandle*)handle;
}

fastKronError fastKronInitCUDA(fastKronHandle handlePtr, void *ptrToStream) {
  return ((FastKronHandle*)handlePtr)->initCUDABackend(ptrToStream, 1, 1, 1, 1);
}

fastKronError fastKronMgInitCUDA(fastKronHandle handlePtr, void *streams, int gpus, int gpusInM, int gpusInK, int gpuLocalKrons) {
  return ((FastKronHandle*)handlePtr)->initCUDABackend(streams, gpus, gpusInM, gpusInK, gpuLocalKrons);
}

fastKronError fastKronInitHIP(fastKronHandle handlePtr, void *ptrToStream) {
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
  KMMProblem problem(FastKronTypeNone,
                     Matrix(M, KMMProblem::getK(Ps, N)), fastKronOp_N,
                     KMMProblem::Factors(N, Ps, Qs, nullptr), fastKronOp_N,
                     Matrix(M, KMMProblem::getL(Qs, N)));
  return ((FastKronHandle*)handlePtr)->gekmmSizes(problem, resultSize, tempSize);
}

fastKronError sgekmm(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], const float* X,
                     fastKronOp opX, const float* Fs[], fastKronOp opFs, float* Y,
                     float alpha, float beta, const float *Z, float* temp1, float* temp2) {
  KMMProblem problem(FastKronFloat,
                     Matrix(M, KMMProblem::getK(Ps, N), (void*)X), opX,
                     KMMProblem::Factors(N, Ps, Qs, (void**)Fs), opFs,
                     Matrix(M, KMMProblem::getL(Qs, N), (void*)Y));
  return ((FastKronHandle*)handle)->xgekmm(problem, backend, (void*)temp1, (void*)temp2,
                        EpilogueParams::create<float>(alpha, beta, Z));
}

fastKronError igekmm(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], const int* X,
                   fastKronOp opX, const int* Fs[], fastKronOp opFs, int* Y,
                   int alpha, int beta, const int *Z, int* temp1, int* temp2) {
  KMMProblem problem(FastKronInt,
                     Matrix(M, KMMProblem::getK(Ps, N), (void*)X), opX,
                     KMMProblem::Factors(N, Ps, Qs, (void**)Fs), opFs,
                     Matrix(M, KMMProblem::getL(Qs, N), (void*)Y));
  return ((FastKronHandle*)handle)->xgekmm(problem, backend, (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<int>(alpha, beta, Z));
}

fastKronError dgekmm(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], const double* X,
                   fastKronOp opX, const double* Fs[], fastKronOp opFs, double* Y,
                   double alpha, double beta, const double *Z, double* temp1, double* temp2) {
  KMMProblem problem(FastKronDouble,
                     Matrix(M, KMMProblem::getK(Ps, N), (void*)X), opX,
                     KMMProblem::Factors(N, Ps, Qs, (void**)Fs), opFs,
                     Matrix(M, KMMProblem::getL(Qs, N), (void*)Y));
  return ((FastKronHandle*)handle)->xgekmm(problem, backend, (void*)temp1, (void*)temp2, 
                        EpilogueParams::create<double>(alpha, beta, Z));
}

fastKronError sgekmmStridedBatched(fastKronHandle handle, fastKronBackend backend, 
                                   uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                                   const float* X, fastKronOp opX, uint64_t strideX,
                                   const float* Fs[], fastKronOp opFs, uint64_t strideF[],
                                   float* Y, float alpha, float beta, uint64_t strideY,
                                   uint32_t batchCount, const float *Z, uint64_t strideZ,
                                   float* temp1, float* temp2) {
  uint32_t K = KMMProblemStridedBatched::getK(Ps, N);
  uint32_t L = KMMProblemStridedBatched::getK(Qs, N);
  KMMProblemStridedBatched::Factor fs[N];
  for (int i = 0; i < N; i++) {
    fs[i] = KMMProblemStridedBatched::Factor(Ps[i], Qs[i], strideF[i], (void*)Fs[i]);
  }
  std::cout << 178 << " " << X << " " << Y << " " << Z << std::endl;
  KMMProblemStridedBatched problem(FastKronFloat, 
          KMMProblemStridedBatched::Matrix(M, K, strideX, (void*)X), opX,
          N, &fs[0], opFs,
          KMMProblemStridedBatched::Matrix(M, L, strideY, (void*)Y), batchCount);

  auto epilogueParams = EpilogueStridedBatchedParams::create<float>(alpha, beta, Z, strideZ);
  return ((FastKronHandle*)handle)->xgekmmStridedBatched(problem, backend, temp1, temp2, epilogueParams);
}

fastKronError igekmmStridedBatched(fastKronHandle handle, fastKronBackend backend,
                     uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                     const int* X, fastKronOp opX, uint64_t strideX,
                     const int* Fs[], fastKronOp opFs, uint64_t strideF[],
                     int* Z, int alpha, int beta, uint64_t strideZ,
                     uint32_t batchCount, const int *Y, uint64_t strideY, int* temp1, int* temp2) {
  
}

fastKronError dgekmmStridedBatched(fastKronHandle handle, fastKronBackend backend,
                     uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                     const double* X, fastKronOp opX, uint64_t strideX,
                     const double* Fs[], fastKronOp opFs, uint64_t strideF[], 
                     double* Z, double alpha, double beta, uint64_t strideZ,
                     uint32_t batchCount, const double *Y, uint64_t strideY, double* temp1, double* temp2) {

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