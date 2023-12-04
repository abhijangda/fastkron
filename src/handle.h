#include <vector>
#include <nccl.h>

#include <unordered_map>

#include "fastkron.h"
#include "thread_pool.h"
#include "kernel_invoker.h"
#include "env.h"
#include "kmmalgo.h"

#pragma once

struct TunedKernelFromStart {
  KernelInfo kernel;
  uint start, end;
  uint K;
  float time;
  bool distShare;

  TunedKernelFromStart(KernelInfo kernel_, uint start_, uint end_, uint K_, float time_):
    kernel(kernel_), start(start_), end(end_), K(K_), time(time_), distShare(false) {}
  TunedKernelFromStart() {}
  friend std::ostream& operator<<(std::ostream &out, const TunedKernelFromStart &k) {
    out << "[" << k.start << ", " << k.end << "] = " << k.K << " " << k.kernel << " runs for " << k.time << " ms";
    return out;
  }
};

template<>
struct std::hash<KronMatmulShape> {
  std::size_t operator()(const KronMatmulShape& k) const;
};

typedef std::vector<TunedKernelFromStart> TunedKernelsSeries;
struct ThreadArgs;

enum ProcType {
  ProcNone = 0,
  SingleThread,
  MultipleThread,
  MultipleProcess
};

struct FastKronHandle {
  void* result_;

  FastKronHandle(int gpus, int gpusInM, int gpusInK, int gpuKrons);
  uint numGPUs_;
  uint gpusInM_;
  uint gpusInK_;
  uint perGPUKronBatch_;
  bool isDistributed_;
  DistComm distComm_;
  KernelInvoker kernelInvoker;
  //Map from Factor size and Number of factors to KernelInfos
  std::unordered_map<KronMatmulShape, std::vector<KernelInfo>> compiledKernels;

  pthread_barrier_t* barriers_;
  thread_pool<ThreadArgs*>* threads_;

  //TODO: these two functions should be a part of utils?
  cudaError_t allocDistributedX(void* dX[], void* hX, uint M, uint K);
  cudaError_t gatherDistributedY(void* dY[], void* hY, uint M, uint K, 
                                 uint NumKronMats, uint KronMatCols[], uint KronMatRows[]);

  void getDistributedSizes(uint M, uint K, uint& gpuM, uint& gpuK);
  void free();

  //Options
  bool useFusion_;
  void setUseFusion(bool v) {useFusion_ = v;}
  bool getUseFusion()       {return useFusion_;}

  TunedKernelsSeries tunedKernelSeries;
  
  std::vector<ncclComm_t> ncclComms;

  KronMatmulShape maxCompiledColsA(KronMatmulShape shape);
  KernelInfo selectKernel(KronMatmulShape shape);
  uint maxFusedKernels(KronMatmulShape shape);

  cudaError_t xgekmm(uint M, uint N, uint Ps[], uint Qs[], void* X, void* Fs[], void* Y,
                     void* temp1, void* temp2, EpilogueParams epilogueParams, cudaStream_t stream);

  cudaError_t distributedsgekmm(const uint NumKronMats, float* x[], float* kronMats[], float* result[],
                                  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
                                  cudaStream_t streams[]);
  cudaError_t fusedDistributedSlicedMatmul(const uint NumFusedKerns, KernelInfo& kernel, const uint kronIndex, 
                                void* x, void** kronMat, void* kronGemmResult,
                                const uint M, const uint N, const uint K, 
                                const uint* FusedKronMatCols, const uint* FusedKronMatRows,
                                DistributedParams distParams, EpilogueParams epilogueParams,
                                cudaStream_t stream);
  cudaError_t fusedSlicedMatmul(uint NumFusedKerns, KernelInfo& kernelInfo, const uint kronIndex, 
                              void* x, void** krons, void* kronGemmResult,
                              const uint M, const uint N, const uint K, 
                              const uint* FusedKronMatCols, const uint* FusedKronMatRows,
                              EpilogueParams epilogueParams,
                              cudaStream_t stream);
  TunedKernelsSeries selectKernelSeries(const uint NumKronMats,
                                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                                      bool distributedKernel);
};

struct ThreadArgs {
  ThreadArgs() {}
  ThreadArgs(FastKronHandle* handle, uint NumKronMats, void* x, void** kronMats, void** result, 
            uint M, uint N, uint K, uint *KronMatCols, uint *KronMatRows, void **temp1,
            void **temp2, cudaStream_t* stream,
            uint gpuRow, uint gpuCol, uint gpusInM_, uint gpusInK_, pthread_barrier_t* barrier) : 
            handle(handle), NumKronMats(NumKronMats), x(x), kronMats(kronMats), result(result),
            M(M), N(N), K(K), KronMatCols(KronMatCols), KronMatRows(KronMatRows), temp1(temp1),
            temp2(temp2), stream(stream),
            gpuRow(gpuRow), gpuCol(gpuCol), gpusInM_(gpusInM_), gpusInK_(gpusInK_), barrier(barrier) {}

  FastKronHandle* handle;
  uint NumKronMats;
  void* x;
  void** kronMats;
  void** result;
  uint M;
  uint N;
  uint K;
  uint *KronMatCols;
  uint *KronMatRows;
  void **temp1;
  void **temp2;
  cudaStream_t* stream;
  uint gpuRow;
  uint gpuCol;
  uint gpusInM_;
  uint gpusInK_;
  pthread_barrier_t* barrier;

  struct ThreadResult {
    cudaError_t status;
    void* result;
  } threadResult;
};

//TODO: Combine all arguments in KronMatmulShape
bool checkDistributedKronSizes(const uint NumKronMats, 
                               const uint M, const uint N, const uint K, 
                               const uint KronMatCols[], const uint KronMatRows[],
                               const uint LocalKrons, const uint gpusInK);
bool checkDistributedKronSizes(const KMMProblem problem,
                               const uint LocalKrons, const uint gpusInK);
bool checkKronMatrixSizes(const uint NumKronMats, 
                          const uint M, const uint N, const uint K, 
                          const uint KronMatCols[], const uint KronMatRows[]);