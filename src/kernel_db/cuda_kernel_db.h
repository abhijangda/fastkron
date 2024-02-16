#include <functional>

#include <cuda_runtime.h>
#include <cuda.h>
#include <nccl.h>

#include "kmm/kmmalgo.h"
#include "kernels/kernel_info.h"
#include "kernels/params.h"
#include "kernel_db/kernel_db.h"
#include "env/env.h"
#include "utils/thread_pool.h"

#pragma once

struct ThreadArgs;

//TODO: Change name to Executor?
class CUDAKernelDatabase : public KernelDatabase {
public:
  cudaStream_t* streams;
  uint numGPUs_;
  uint gpusInM_;
  uint gpusInK_;
  uint perGPUKronBatch_;
  bool isDistributed_;
  DistComm distComm_;
  std::vector<ncclComm_t> ncclComms;
  pthread_barrier_t* barriers_;
  thread_pool<ThreadArgs*>* threads_;

public:
  CUDAKernelDatabase();

  cudaError_t init(void* ptrToStream, int gpus, int gpusInM, int gpusInK, int gpuKrons);

  void free() {
    compiledKernels.clear();
    delete[] streams;
    if (isDistributed_) {
      for (uint g = 0; g < gpusInM_; g++) {
        int s = pthread_barrier_destroy(&barriers_[g]);
        PTHREAD_BARRIER_CHECK(s);
      }

      delete threads_;
      delete barriers_;

      if (distComm_ == DistComm::NCCL) {
        for (int i=0; i<ncclComms.size(); i++)
          ncclCommDestroy(ncclComms[i]);
      }
    }
  }
  
  virtual cudaError_t invokeKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                   KMMProblem problem,
                                   EpilogueParams epilogueParams,
                                   KernelMode execMode);
  virtual cudaError_t invokeP2PStoreKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                           KMMProblem problem, DistributedParams distParams, 
                                           EpilogueParams epilogueParams,
                                           KernelMode execMode);

  std::pair<KernelInfo, float> tuneKernelForProblem(KMMProblem problem, bool distP2PStore, uint factorIdx, DistributedParams distParams);
  virtual cudaError_t procMalloc(uint32_t proc, size_t size, void*& ptr);
  virtual cudaError_t procMalloc(uint32_t proc, Matrix& m);
  virtual cudaError_t procMemset(uint32_t proc, Matrix& m, float val);
  virtual cudaError_t procFree(uint32_t proc, Matrix m);
  virtual cudaError_t procFree(uint32_t proc, void* ptr);
};