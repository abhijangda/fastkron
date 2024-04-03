#include <functional>
#include <vector>
#include <map>

#include "kmm/kmmalgo.h"
#include "kernels/params.h"
#include "kernel_db/kernel_db.h"
#include "env/env.h"
#include "utils/thread_pool.h"

#pragma once

struct ThreadArgs;

class OptimizedKernelForShape {
  std::map<KMMProblem, KernelInfo*, KMMProblemComparator> shapeToKernel;
public:
  void init(KernelDatabase* db, std::unordered_map<KMMProblem, std::string> shapeToKernelStr) {
    for (auto iter : shapeToKernelStr) {
      shapeToKernel[iter.first] = db->getKernel(iter.second);
    }
  }
};

class CUDAArchDetail {
public:
  uint32_t numSMs;
  uint32_t maxBlocksPerSM;
  uint32_t maxThreadsPerBlock;
  uint32_t maxThreadsPerSM;
  uint32_t regsPerSM;
  uint32_t maxRegsPerThread;
  uint32_t sharedMemPerSM;
  uint32_t sharedMemPerBlock;
  std::string name;
  uint32_t computeMajor;
  uint32_t computeMinor;


  // CUDAArchDetail(uint32_t numSMs, uint32_t maxBlocksPerSM, uint32_t maxThreadsPerBlock,
  //                uint32_t maxThreadsPerSM, uint32_t regsPerSM, uint32_t sharedMemPerSM) :
  //                numSMs(numSMs), maxBlocksPerSM(maxBlocksPerSM), 
  //                maxThreadsPerBlock(maxThreadsPerBlock),
  //                maxThreadsPerSM(maxThreadsPerSM), 
  //                regsPerSM(regsPerSM), sharedMemPerSM(sharedMemPerSM) {}
  CUDAArchDetail(int dev);
  
  friend std::ostream& operator<<(std::ostream &out, const CUDAArchDetail &detail) {
    std::string indent = "    ";
    out << detail.name << std::endl <<
          indent << "Compute Capability      : " << (detail.computeMajor*10 + detail.computeMinor) << std::endl <<
          indent << "SMs                     : " << detail.numSMs       << std::endl <<
          indent << "Max Blocks per SM       : " << detail.maxBlocksPerSM << std::endl <<
          indent << "Max Threads per SM      : " << detail.maxThreadsPerSM << std::endl <<
          indent << "Registers Per SM        : " << detail.regsPerSM << std::endl <<
          indent << "Shared Memory per SM    : " << detail.sharedMemPerSM << std::endl<<
          indent << "Shared Memory Per Block : " << detail.sharedMemPerBlock << std::endl;

    return out;
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
  std::vector<CUDAArchDetail> gpusDetail;
  OptimizedKernelForShape fastestKernelForShape;

public:
  CUDAKernelDatabase();
  ~CUDAKernelDatabase() {}

  fastKronError init(void* ptrToStream, int gpus, int gpusInM, int gpusInK, int gpuKrons);
  static int numDevices();
  CUDAArchDetail parseDeviceProperties(int dev);

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
  virtual KernelInfo* kernelForSubProblem(KMMProblem subProblem);
  virtual fastKronError procMalloc(uint32_t proc, size_t size, void*& ptr);
  virtual fastKronError procMemset(uint32_t proc, Matrix& m, float val);
  virtual fastKronError procFree(uint32_t proc, void* ptr);
};