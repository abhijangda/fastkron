#include "kernel_db.h"

#include "kmm/kmmalgo.h"
#include "kernels/kmmkernel.h"
#include "kernels/params.h"
#include "kernel_db/kernel_db.h"
#include "env/env.h"
#include "utils/thread_pool.h"

class HIPKernelDatabase : public KernelDatabase {
public:
  std::vector<void*> streams;
  // uint numGPUs_;
  // uint gpusInM_;
  // uint gpusInK_;
  // uint perGPUKronBatch_;
  // bool isDistributed_;
  // DistComm distComm_;
  // std::vector<ncclComm_t> ncclComms;
  // pthread_barrier_t* barriers_;
  // thread_pool<ThreadArgs*>* threads_;

public:
  HIPKernelDatabase();
  ~HIPKernelDatabase() {}

  fastKronError init(void* ptrToStream){
    streams.clear();
    streams.push_back(ptrToStream);
    return fastKronSuccess;
  }
  
  void free() {
    streams.clear();
    // if (isDistributed_) {
    //   for (uint g = 0; g < gpusInM_; g++) {
    //     int s = pthread_barrier_destroy(&barriers_[g]);
    //     PTHREAD_BARRIER_CHECK(s);
    //   }

    //   delete threads_;
    //   delete barriers_;

    //   if (distComm_ == DistComm::NCCL) {
    //     for (int i=0; i<ncclComms.size(); i++)
    //       ncclCommDestroy(ncclComms[i]);
    //   }
    // }
  }

  virtual fastKronError invokeKernel(KMMKernel* kernelInfo, const uint kronIndex, 
                                     KMMProblem problem,
                                     EpilogueParams epilogueParams,
                                     KernelMode execMode);
  virtual fastKronError invokeP2PStoreKernel(KMMKernel* kernelInfo, const uint kronIndex, 
                                           KMMProblem problem, DistributedParams distParams, 
                                           EpilogueParams epilogueParams,
                                           KernelMode execMode){return fastKronSuccess;}
  virtual fastKronError timeKernel(KMMKernel* kernelInfo, const uint kronIndex, 
                                 KMMProblem problem, DistributedParams distParams, 
                                 EpilogueParams epilogueParams,
                                 KernelMode execMode, 
                                 bool distP2PStore,
                                 int warmups, int runs,
                                 float& runtime);
  virtual fastKronError procMalloc(uint32_t proc, size_t size, void*& ptr);
  virtual fastKronError procMemset(uint32_t proc, Matrix& m, float val);
  virtual fastKronError procFree(uint32_t proc, void* ptr);
};