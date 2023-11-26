#include <vector>
#include <nccl.h>

#include "fastkron.h"
#include "thread_pool.h"
#include "device/kernel_info.h"
#include "device/params.h"
#include "env.h"

#pragma once

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

struct TunedKernelFromStart {
  KernelInfo kernel;
  uint start, end;
  uint K;
  float time;

  TunedKernelFromStart(KernelInfo kernel_, uint start_, uint end_, uint K_, float time_):
    kernel(kernel_), start(start_), end(end_), K(K_), time(time_) {}
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

  FastKronHandle() :
    tunedKernelSeries()
  {
    //Optimization Options
    useFusion_ = true;

    //Is Distributed
    isDistributed_ = false;
  }

  uint numGPUs_;
  uint gpusInM_;
  uint gpusInK_;
  uint perGPUKronBatch_;
  bool isDistributed_;
  DistComm distComm_;
  //Map from Factor size and Number of factors to KernelInfos
  std::unordered_map<KronMatmulShape, std::vector<KernelInfo>> compiledKernels;

  pthread_barrier_t* barriers_;
  thread_pool<ThreadArgs*>* threads_;

  //TODO: these two functions should be a part of utils?
  template<typename T> cudaError_t allocDistributedX(T* dX[], T* hX, uint M, uint K);
  template<typename T> cudaError_t gatherDistributedY(T* dY[], T* hY, uint M, uint K, 
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

  cudaError_t sgekmm(const uint NumKronMats, float* x, float* kronMats[], 
  float* result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], 
  float* temp1, float* temp2, 
  EpilogueParams<float> epilogueParams,
  cudaStream_t stream);

  cudaError_t igekmm(const uint NumKronMats, int* x, int* kronMats[],
                                  int* result,
                                  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], 
                                  int* temp1, int* temp2, 
                                  EpilogueParams<int> epilogueParams,
                                  cudaStream_t stream);

  cudaError_t distributedsgekmm(const uint NumKronMats, float* x[], float* kronMats[], float* result[],
                                  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
                                  cudaStream_t streams[]);
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

struct Autotuner {
  cudaError_t tune(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], 
                     uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                     cudaStream_t stream);

  cudaError_t tune(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], 
                     uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                     cudaStream_t stream);

  cudaError_t tune(FastKronHandle& handle, const uint NumKronMats, double* x, double* kronMats[], 
                     uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                     cudaStream_t stream);  
};

//TODO: Combine all arguments in KronMatmulShape
bool checkDistributedKronSizes(const uint NumKronMats, 
                               const uint M, const uint N, const uint K, 
                               const uint KronMatCols[], const uint KronMatRows[],
                               const uint LocalKrons, const uint gpusInK);
bool checkKronMatrixSizes(const uint NumKronMats, 
                          const uint M, const uint N, const uint K, 
                          const uint KronMatCols[], const uint KronMatRows[]);