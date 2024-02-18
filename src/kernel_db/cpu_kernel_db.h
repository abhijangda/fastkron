#include <functional>
#include <vector>

#include "kernel_db/kernel_db.h"

class CPUKernelDatabase : public KernelDatabase {

public:
  CPUKernelDatabase() : KernelDatabase() {}

  virtual cudaError_t invokeKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                   KMMProblem problem,
                                   EpilogueParams epilogueParams,
                                   KernelMode execMode) {}
  virtual cudaError_t invokeP2PStoreKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                           KMMProblem problem, DistributedParams distParams, 
                                           EpilogueParams epilogueParams,
                                           KernelMode execMode) {}
  virtual cudaError_t timeKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                 KMMProblem problem, DistributedParams distParams, 
                                 EpilogueParams epilogueParams,
                                 KernelMode execMode, 
                                 bool distP2PStore,
                                 int warmups, int runs,
                                 float& runtime) {}
  virtual cudaError_t procMalloc(uint32_t proc, size_t size, void*& ptr){}
  virtual cudaError_t procMalloc(uint32_t proc, Matrix& m){}
  virtual cudaError_t procMemset(uint32_t proc, Matrix& m, float val){}
  virtual cudaError_t procFree(uint32_t proc, Matrix m){}
  virtual cudaError_t procFree(uint32_t proc, void* ptr){}
};

class X86KernelDatabase : public CPUKernelDatabase {
};