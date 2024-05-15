#include <functional>
#include <vector>
#include <unistd.h>

#include "kernel_db/kernel_db.h"

struct CPUCache {
  uint32_t threads;
  uint32_t size;
  void** ptr;

  CPUCache() : threads(0), size(0), ptr(nullptr) {}

  void alloc(uint32_t threads, uint32_t size) {
    this->threads = threads;
    this->size = size;
    ptr = (void**)malloc(threads * sizeof(void*));
    uint32_t pageSize = getpagesize();
    for (int i = 0; i < threads; i++) {
      ptr[i] = aligned_alloc(pageSize, size);
    }
  }

  ~CPUCache() {
    for (int i = 0; i < threads; i++) {
      free(ptr[i]);
    }
    free(ptr);
    ptr = nullptr;
    threads = 0;
    size = 0;
  }
};

class CPUKernelDatabase : public KernelDatabase {
protected:
  char* trash1, *trash2;
  CPUCache TileXs;
  CPUCache TileYs;
  CPUCache TileFs;

public:
  CPUKernelDatabase();

  void init() {}
  void allocate_caches();
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

  uint32_t getMaxThreads() const {
    return MAX(omp_get_max_threads(), getCPUProperties().sockets * getCPUProperties().cores);
  }
  CPUArchDetails getCPUProperties() const {return *(dynamic_cast<const CPUArchDetails*>(hardware[0]));}
  void free();
};

class X86KernelDatabase : public CPUKernelDatabase {
public:
  X86KernelDatabase();
  X86ArchDetails getX86CPUProperties() const {return *(dynamic_cast<const X86ArchDetails*>(hardware[0]));}
  virtual KernelInfo* kernelForSubProblem(KMMProblem subProblem, const std::vector<KernelInfo*>& kernels);
};