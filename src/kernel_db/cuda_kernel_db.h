#include <functional>
#include <vector>
#include <map>

#include "kmm/kmmalgo.h"
#include "kernels/params.h"
#include "kernel_db/kernel_db.h"
#include "env/env.h"
#include "utils/thread_pool.h"
#include "kernels/hw_details.h"

#pragma once

struct ThreadArgs;

//TODO: Change name to Executor?
class CUDAKernelDatabase : public KernelDatabase {
public:
  std::vector<void*> streams;
  uint numGPUs_;
  uint gpusInM_;
  uint gpusInK_;
  uint perGPUKronBatch_;
  bool isDistributed_;
  DistComm distComm_;
  std::vector<void*> ncclComms;
  pthread_barrier_t* barriers_;
  thread_pool<ThreadArgs*>* threads_;

public:
  CUDAKernelDatabase();
  ~CUDAKernelDatabase();

  fastKronError init(void* ptrToStream, int gpus, int gpusInM, int gpusInK, int gpuKrons);
  static int numDevices();
  CUDAArchDetails parseDeviceProperties(int dev);
  CUDAArchDetails getCUDADeviceProperties() {return *(dynamic_cast<CUDAArchDetails*>(hardware[0]));}
  void setCUDAStream(void* ptrToStream);
  virtual fastKronError initTune();
  virtual fastKronError invokeKernel(KernelInfo* kernelInfo, const uint kronIndex, 
                                   KMMProblem problem,
                                   EpilogueParams epilogueParams,
                                   KernelMode execMode);
  virtual fastKronError invokeP2PStoreKernel(KernelInfo* kernelInfo, const uint kronIndex, 
                                           KMMProblem problem, DistributedParams distParams, 
                                           EpilogueParams epilogueParams,
                                           KernelMode execMode);
  virtual fastKronError timeKernel(KernelInfo* kernelInfo, const uint kronIndex, 
                                 KMMProblem problem, DistributedParams distParams, 
                                 EpilogueParams epilogueParams,
                                 KernelMode execMode, 
                                 bool distP2PStore,
                                 int warmups, int runs,
                                 float& runtime);

  virtual std::string   occupancyDetails(KernelInfo* kernelInfo, KMMProblem problem);
  virtual std::map<uint32_t, std::vector<KernelInfo*>, std::greater<int>> filterFastestFusedKernels(const KMMProblem& problem, const std::vector<KernelInfo*>& kernels);
  virtual KernelInfo* kernelForSubProblem(KMMProblem subProblem, const std::vector<KernelInfo*>& kernels);
  virtual fastKronError procMalloc(uint32_t proc, size_t size, void*& ptr);
  virtual fastKronError procMemset(uint32_t proc, Matrix& m, float val);
  virtual fastKronError procFree(uint32_t proc, void* ptr);
};