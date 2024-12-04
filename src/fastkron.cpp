#include <cassert>

#include <iostream>
#include <unordered_map>
#include <utility>

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
  if (env::getUseTune()) fastKronOptionsAll |= fastKronOptionsTune;

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
  KMMProblem problem(FastKronMMType::MKM, FastKronTypeNone,
                     Matrix(M, KMMProblem::getK(Ps, N)), fastKronOp_N,
                     KMMProblem::Factors(N, Ps, Qs, nullptr), fastKronOp_N,
                     Matrix(M, KMMProblem::getL(Qs, N)));
  return ((FastKronHandle*)handlePtr)->gekmmSizes(problem, resultSize, tempSize);
}

fastKronError gekmmSizesForward(fastKronHandle handlePtr, uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                                size_t* resultSize, size_t* intermediateSizes) {
  KMMProblem problem(FastKronMMType::MKM, FastKronTypeNone,
                     Matrix(M, KMMProblem::getK(Ps, N)), fastKronOp_N,
                     KMMProblem::Factors(N, Ps, Qs, nullptr), fastKronOp_N,
                     Matrix(M, KMMProblem::getL(Qs, N)));
  return ((FastKronHandle*)handlePtr)->gekmmSizesForward(problem, resultSize, intermediateSizes);
}

fastKronError sgemkm(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], const float* X,
                     fastKronOp opX, const float* const Fs[], fastKronOp opFs, float* Y,
                     float alpha, float beta, const float *Z, float* temp1, float* temp2) {
  KMMProblem problem(FastKronMMType::MKM, FastKronFloat,
                     Matrix(M, KMMProblem::getK(Ps, N), (void*)X), opX,
                     KMMProblem::Factors(N, Ps, Qs, (void**)Fs), opFs,
                     Matrix(M, KMMProblem::getL(Qs, N), (void*)Y));
  return ((FastKronHandle*)handle)->xgemm(problem, backend, (void*)temp1, (void*)temp2,
                                           EpilogueParams::create<float>(alpha, beta, Z));
}

fastKronError dgemkm(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], const double* X,
                   fastKronOp opX, const double* const Fs[], fastKronOp opFs, double* Y,
                   double alpha, double beta, const double *Z, double* temp1, double* temp2) {
  KMMProblem problem(FastKronMMType::MKM, FastKronDouble,
                     Matrix(M, KMMProblem::getK(Ps, N), (void*)X), opX,
                     KMMProblem::Factors(N, Ps, Qs, (void**)Fs), opFs,
                     Matrix(M, KMMProblem::getL(Qs, N), (void*)Y));
  return ((FastKronHandle*)handle)->xgemm(problem, backend, (void*)temp1, (void*)temp2,
                                          EpilogueParams::create<double>(alpha, beta, Z));
}

fastKronError sgekmm(fastKronHandle handle, fastKronBackend backend, 
                     uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                     const float* const Fs[], fastKronOp opFs,
                     const float* X, fastKronOp opX,
                     float* Y, float alpha, float beta,
                     const float *Z, float* temp1, float* temp2) {
  KMMProblem problem(FastKronMMType::KMM, FastKronFloat,
                     Matrix(M, KMMProblem::getK(Ps, N), (void*)X), opX,
                     KMMProblem::Factors(N, Ps, Qs, (void**)Fs), opFs,
                     Matrix(M, KMMProblem::getL(Qs, N), (void*)Y));
  return ((FastKronHandle*)handle)->xgemm(problem, backend, (void*)temp1, (void*)temp2, 
                                           EpilogueParams::create<float>(alpha, beta, Z));
}

fastKronError dgekmm(fastKronHandle handle, fastKronBackend backend, 
                     uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                     const double* const Fs[], fastKronOp opFs,
                     const double* X, fastKronOp opX,
                     double* Y, double alpha, double beta,
                     const double *Z, double* temp1, double* temp2) {
  KMMProblem problem(FastKronMMType::KMM, FastKronDouble,
                     Matrix(M, KMMProblem::getK(Ps, N), (void*)X), opX,
                     KMMProblem::Factors(N, Ps, Qs, (void**)Fs), opFs,
                     Matrix(M, KMMProblem::getL(Qs, N), (void*)Y));
  return ((FastKronHandle*)handle)->xgemm(problem, backend, (void*)temp1, (void*)temp2, 
                                           EpilogueParams::create<double>(alpha, beta, Z));
}

template<typename ElemT>
std::pair<KMMProblemStridedBatched, EpilogueStridedBatchedParams> 
  createStridedBatchedProblem(FastKronMMType mmtype, FastKronType type, 
                              uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                              const ElemT* X, fastKronOp opX, uint64_t strideX,
                              const ElemT* const Fs[], fastKronOp opFs, uint64_t strideF[],
                              ElemT* Y, uint64_t strideY, const ElemT alpha, const ElemT beta,
                              uint32_t batchCount, const ElemT *Z, uint64_t strideZ) {
  uint32_t K = KMMProblemStridedBatched::getK(Ps, N);
  uint32_t L = KMMProblemStridedBatched::getK(Qs, N);
  KMMProblemStridedBatched::Factor fs[N];
  for (uint32_t i = 0; i < N; i++) {
    fs[i] = KMMProblemStridedBatched::Factor(Ps[i], Qs[i], strideF[i], (void*)Fs[i]);
  }

  KMMProblemStridedBatched problem(mmtype, type, 
          KMMProblemStridedBatched::Matrix(M, K, strideX, (void*)X), opX,
          N, &fs[0], opFs,
          KMMProblemStridedBatched::Matrix(M, L, strideY, (void*)Y), batchCount);

  auto epilogueParams = EpilogueStridedBatchedParams::create<ElemT>(alpha, beta, 
                            StridedBatchMatrix(M, L, strideZ, (void*)Z));

  return std::make_pair(problem, epilogueParams);
}

fastKronError sgemkmStridedBatched(fastKronHandle handle, fastKronBackend backend, 
                                   uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                                   const float* X, fastKronOp opX, uint64_t strideX,
                                   const float* const Fs[], fastKronOp opFs, uint64_t strideF[],
                                   float* Y, uint64_t strideY, float alpha, float beta,
                                   uint32_t batchCount, const float *Z, uint64_t strideZ,
                                   float* temp1, float* temp2) {
  auto problem = createStridedBatchedProblem(FastKronMMType::MKM, FastKronFloat,
                                             M, N, Ps, Qs, X, opX, strideX, Fs, opFs, strideF, 
                                             Y, strideY, alpha, beta, batchCount, Z, strideZ);
  return ((FastKronHandle*)handle)->xgemmStridedBatched(std::get<0>(problem), backend,
                                                         temp1, temp2,
                                                         std::get<1>(problem));
}

fastKronError dgemkmStridedBatched(fastKronHandle handle, fastKronBackend backend,
                     uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                     const double* X, fastKronOp opX, uint64_t strideX,
                     const double* const Fs[], fastKronOp opFs, uint64_t strideF[], 
                     double* Y, uint64_t strideY, double alpha, double beta,
                     uint32_t batchCount, const double *Z, uint64_t strideZ, double* temp1, double* temp2) {
  auto problem = createStridedBatchedProblem(FastKronMMType::MKM, FastKronDouble,
                                             M, N, Ps, Qs, X, opX, strideX, Fs, opFs, strideF,
                                             Y, strideY, alpha, beta, batchCount, Z, strideZ);
  return ((FastKronHandle*)handle)->xgemmStridedBatched(std::get<0>(problem), backend,
                                                        temp1, temp2,
                                                        std::get<1>(problem));
}

fastKronError sgekmmStridedBatched(fastKronHandle handle, fastKronBackend backend, 
                     uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                     const float* const Fs[], fastKronOp opFs, uint64_t strideF[],
                     const float* X, fastKronOp opX, uint64_t strideX,
                     float* Y, uint64_t strideY, float alpha, float beta,
                     uint32_t batchCount, const float *Z, uint64_t strideZ, 
                     float* temp1, float* temp2) {
  auto problem = createStridedBatchedProblem(FastKronMMType::KMM, FastKronFloat,
                                             M, N, Ps, Qs, X, opX, strideX, Fs, opFs, strideF, 
                                             Y, strideY, alpha, beta, batchCount, Z, strideZ);
  return ((FastKronHandle*)handle)->xgemmStridedBatched(std::get<0>(problem), backend,
                                                        temp1, temp2,
                                                        std::get<1>(problem));
}

fastKronError dgekmmStridedBatched(fastKronHandle handle, fastKronBackend backend,
                     uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                     const double* const Fs[], fastKronOp opFs, uint64_t strideF[],
                     const double* X, fastKronOp opX, uint64_t strideX, 
                     double* Y, uint64_t strideY, double alpha, double beta,
                     uint32_t batchCount, const double *Z, uint64_t strideZ,
                     double* temp1, double* temp2) {
  auto problem = createStridedBatchedProblem(FastKronMMType::KMM, FastKronDouble,
                                             M, N, Ps, Qs, X, opX, strideX, Fs, opFs, strideF,
                                             Y, strideY, alpha, beta, batchCount, Z, strideZ);
  return ((FastKronHandle*)handle)->xgemmStridedBatched(std::get<0>(problem), backend,
                                                         temp1, temp2,
                                                         std::get<1>(problem));
}

fastKronError smkmForward(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], const float* X,
                          fastKronOp opX, const float* const Fs[], fastKronOp opFs, float* Y,
                          float* Intermediates[]) {
  KMMProblem problem(FastKronMMType::MKM, FastKronFloat,
                     Matrix(M, KMMProblem::getK(Ps, N), (void*)X), opX,
                     KMMProblem::Factors(N, Ps, Qs, (void**)Fs), opFs,
                     Matrix(M, KMMProblem::getL(Qs, N), (void*)Y));
  return ((FastKronHandle*)handle)->xgemm(true, problem, backend, (void**)Intermediates,
                                          EpilogueParams::create<float>(1.0f, 0.0f, nullptr));
}

fastKronError dmkmForward(fastKronHandle handle, fastKronBackend backend, uint M, uint N, uint Ps[], uint Qs[], const double* X,
                          fastKronOp opX, const double* const Fs[], fastKronOp opFs, double* Y,
                          double* Intermediates[]) {
  KMMProblem problem(FastKronMMType::MKM, FastKronDouble,
                     Matrix(M, KMMProblem::getK(Ps, N), (void*)X), opX,
                     KMMProblem::Factors(N, Ps, Qs, (void**)Fs), opFs,
                     Matrix(M, KMMProblem::getL(Qs, N), (void*)Y));
  
  return ((FastKronHandle*)handle)->xgemm(true, problem, backend, (void**)Intermediates,
                                          EpilogueParams::create<double>(1.0, 0.0, nullptr));
}

fastKronError skmmForward(fastKronHandle handle, fastKronBackend backend, 
                            uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                            const float* const Fs[], fastKronOp opFs,
                            const float* X, fastKronOp opX,
                            float* Y, float* Intermediates[]) {
  KMMProblem problem(FastKronMMType::KMM, FastKronFloat,
                     Matrix(M, KMMProblem::getK(Ps, N), (void*)X), opX,
                     KMMProblem::Factors(N, Ps, Qs, (void**)Fs), opFs,
                     Matrix(M, KMMProblem::getL(Qs, N), (void*)Y));
  return ((FastKronHandle*)handle)->xgemm(true, problem, backend, (void**)Intermediates,
                                          EpilogueParams::create<float>(1.0f, 0.0f, nullptr));
}

fastKronError dkmmForward(fastKronHandle handle, fastKronBackend backend, 
                            uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                            const double* const Fs[], fastKronOp opFs,
                            const double* X, fastKronOp opX,
                            double* Y, double* Intermediates[]) {
  KMMProblem problem(FastKronMMType::KMM, FastKronDouble,
                     Matrix(M, KMMProblem::getK(Ps, N), (void*)X), opX,
                     KMMProblem::Factors(N, Ps, Qs, (void**)Fs), opFs,
                     Matrix(M, KMMProblem::getL(Qs, N), (void*)Y));
  return ((FastKronHandle*)handle)->xgemm(true, problem, backend, (void**)Intermediates,
                                          EpilogueParams::create<double>(1.0, 0.0, nullptr));
}

fastKronError smkmForwardStridedBatched(fastKronHandle handle, fastKronBackend backend, 
                                          uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                                          const float* X, fastKronOp opX, uint64_t strideX,
                                          const float* const Fs[], fastKronOp opFs, uint64_t strideF[],
                                          float* Y, uint64_t strideY, uint32_t batchCount,
                                          float* Intermediates[], uint64_t strideIntermediates[]) {
  auto problem = createStridedBatchedProblem(FastKronMMType::MKM, FastKronFloat,
                                             M, N, Ps, Qs, X, opX, strideX, Fs, opFs, strideF, 
                                             Y, strideY, 1.0f, 0.0f,
                                             batchCount, (const float*)nullptr, 0);
  return ((FastKronHandle*)handle)->xgemmStridedBatched(true, std::get<0>(problem), backend,
                                                        (void**)Intermediates, strideIntermediates,
                                                        std::get<1>(problem));
}

fastKronError dmkmForwardStridedBatched(fastKronHandle handle, fastKronBackend backend,
                                          uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                                          const double* X, fastKronOp opX, uint64_t strideX,
                                          const double* const Fs[], fastKronOp opFs, uint64_t strideF[], 
                                          double* Y, uint64_t strideY, uint32_t batchCount,
                                          double* Intermediates[], uint64_t strideIntermediates[]) {
  auto problem = createStridedBatchedProblem(FastKronMMType::MKM, FastKronDouble,
                                             M, N, Ps, Qs, X, opX, strideX, Fs, opFs, strideF,
                                             Y, strideY, 1.0, 0.0, batchCount,
                                             (const double*)nullptr, 0);
  return ((FastKronHandle*)handle)->xgemmStridedBatched(true, std::get<0>(problem), backend,
                                                        (void**)Intermediates, strideIntermediates,
                                                        std::get<1>(problem));
}

fastKronError skmmForwardStridedBatched(fastKronHandle handle, fastKronBackend backend, 
                                          uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                                          const float* const Fs[], fastKronOp opFs, uint64_t strideF[],
                                          const float* X, fastKronOp opX, uint64_t strideX,
                                          float* Y, uint64_t strideY, uint32_t batchCount,
                                          float* Intermediates[], uint64_t strideIntermediates[]) {
  auto problem = createStridedBatchedProblem(FastKronMMType::KMM, FastKronFloat,
                                             M, N, Ps, Qs, X, opX, strideX, Fs, opFs, strideF, 
                                             Y, strideY, 1.0f, 0.0f, batchCount,
                                             (const float*)nullptr, 0);
  return ((FastKronHandle*)handle)->xgemmStridedBatched(true, std::get<0>(problem), backend,
                                                        (void**)Intermediates, strideIntermediates,
                                                        std::get<1>(problem));
}

fastKronError dkmmForwardStridedBatched(fastKronHandle handle, fastKronBackend backend,
                                          uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                                          const double* const Fs[], fastKronOp opFs, uint64_t strideF[],
                                          const double* X, fastKronOp opX, uint64_t strideX, 
                                          double* Y, uint64_t strideY, uint32_t batchCount,
                                          double* Intermediates[], uint64_t strideIntermediates[]) {
  auto problem = createStridedBatchedProblem(FastKronMMType::KMM, FastKronDouble,
                                             M, N, Ps, Qs, X, opX, strideX, Fs, opFs, strideF,
                                             Y, strideY, 1.0, 0.0, batchCount,
                                             (const double*)nullptr, 0);
  return ((FastKronHandle*)handle)->xgemmStridedBatched(true, std::get<0>(problem), backend,
                                                        (void**)Intermediates, strideIntermediates,
                                                        std::get<1>(problem));
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