#include <functional>
#include <vector>

#include "kernel_db/kernel_db.h"

class CPUKernelDatabase : public KernelDatabase {
protected:
  char* trash1, *trash2;

public:
  CPUKernelDatabase();

  void init() {}
  virtual fastKronError initTune() {return fastKronSuccess;}
  virtual fastKronError invokeKernel(KernelInfo* kernelInfo, const uint kronIndex, 
                                   KMMProblem problem,
                                   EpilogueParams epilogueParams,
                                   KernelMode execMode);
  virtual fastKronError invokeP2PStoreKernel(KernelInfo* kernelInfo, const uint kronIndex, 
                                           KMMProblem problem, DistributedParams distParams, 
                                           EpilogueParams epilogueParams,
                                           KernelMode execMode) {return fastKronSuccess;}
  virtual fastKronError timeKernel(KernelInfo* kernelInfo, const uint kronIndex, 
                                 KMMProblem problem, DistributedParams distParams, 
                                 EpilogueParams epilogueParams,
                                 KernelMode execMode, 
                                 bool distP2PStore,
                                 int warmups, int runs,
                                 float& runtime);
  virtual std::string   occupancyDetails(KernelInfo* kernelInfo, KMMProblem problem) {return "";}
  virtual fastKronError procMalloc(uint32_t proc, size_t size, void*& ptr);
  virtual fastKronError procMemset(uint32_t proc, Matrix& m, float val);
  virtual fastKronError procFree(uint32_t proc, void* ptr);
  virtual KernelInfo* kernelForSubProblem(KMMProblem subProblem, const std::vector<KernelInfo*>& kernels);

  uint32_t getMaxThreads() const {
    return MAX(omp_get_max_threads(), getCPUProperties().sockets * getCPUProperties().cores);
  }
  CPUArchDetails getCPUProperties() const {return *(dynamic_cast<const CPUArchDetails*>(hardware[0]));}
  void free();
};

class X86KernelDatabase : public CPUKernelDatabase {
public:
  X86KernelDatabase();
};