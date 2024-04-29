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


template<>
struct std::hash<std::pair<Factor, uint32_t>> {
  std::size_t operator()(const std::pair<Factor, uint32_t>& m) const;
};

class OptimizedKernelForShape {
  std::unordered_map<std::pair<Factor, uint>, std::map<Matrix, KernelInfo*, MatrixComparator>> shapeToKernel;
public:
  void init(KernelDatabase* db, std::unordered_map<std::pair<Factor, uint>, std::map<Matrix, std::string, MatrixComparator>> shapeToKernelStr) {
    for (auto factorIter : shapeToKernelStr) {
      shapeToKernel[factorIter.first] = {};
      for (auto iter : factorIter.second) {
        shapeToKernel[factorIter.first][iter.first] = db->getKernel(iter.second);
      }
    }

    if (true) {
      for (auto factorIter : shapeToKernel) {
        for (auto iter : factorIter.second) {
          std::cout << iter.second->str() << std::endl;
        }
      }
    }
  }
};

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
  OptimizedKernelForShape fastestKernelForShape;
  std::unordered_map<KMMProblem, TunedKernelsSeries> problemToKernelCache;

public:
  CUDAKernelDatabase();
  ~CUDAKernelDatabase() {}

  fastKronError init(void* ptrToStream, int gpus, int gpusInM, int gpusInK, int gpuKrons);
  static int numDevices();
  CUDAArchDetails parseDeviceProperties(int dev);
  CUDAArchDetails getCUDADeviceProperties() {return *(dynamic_cast<CUDAArchDetails*>(hardware[0]));}
  void free();
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
  virtual TunedKernelsSeries kernelSeriesForProblem(KMMProblem problem);
  virtual std::map<uint32_t, std::vector<KernelInfo*>, std::greater<int>> filterFastestFusedKernels(const KMMProblem& problem, const std::vector<KernelInfo*>& kernels);
  virtual KernelInfo* kernelForSubProblem(KMMProblem subProblem, const std::vector<std::vector<KernelInfo*>>& kernels);
  virtual KernelInfo* kernelForSubProblem(KMMProblem subProblem, const std::vector<KernelInfo*>& kernels);
  virtual fastKronError procMalloc(uint32_t proc, size_t size, void*& ptr);
  virtual fastKronError procMemset(uint32_t proc, Matrix& m, float val);
  virtual fastKronError procFree(uint32_t proc, void* ptr);
};