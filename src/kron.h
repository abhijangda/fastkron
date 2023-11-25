#include <vector>
#include <nccl.h>

#include "fastkron.h"
#include "thread_pool.h"
#include "device/kernel_info.h"
#include "device/params.h"

#ifndef __KRON_H__
#define __KRON_H__


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

typedef std::vector<TunedKernelFromStart> TunedKernelsSeries;
struct ThreadArgs;
enum DistComm {
  DistCommNone = 0,
  P2P,
  NCCL,
};

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

template<typename T, typename VecT>
cudaError_t singleGPUKronMatmul(FastKronHandle& handle, const uint NumKronMats, T* x, T* kronMats[], 
                                T* result,
                                uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], 
                                T* temp1, T* temp2, 
                                EpilogueParams<T> epilogueParams,
                                cudaStream_t stream);

template<typename T, typename VecT>
cudaError_t distributedKronMatmul(FastKronHandle& handle, const uint NumKronMats, T* x[], T* kronMats[], T* result[],
                                  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
                                  cudaStream_t streams[]);
template<typename T>
cudaError_t autotune(FastKronHandle& handle, const uint NumKronMats, T* x, T* kronMats[], 
                     uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                     cudaStream_t stream);
bool checkDistributedKronSizes(const uint NumKronMats, 
                                      const uint M, const uint N, const uint K, 
                                      const uint KronMatCols[], const uint KronMatRows[],
                                      const uint LocalKrons, const uint gpusInK);
bool checkKronMatrixSizes(const uint NumKronMats, 
                          const uint M, const uint N, const uint K, 
                          const uint KronMatCols[], const uint KronMatRows[]);

#endif