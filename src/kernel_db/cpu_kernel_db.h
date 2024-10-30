#include <functional>
#include <vector>
#include <unistd.h>

#include "kernel_db/kernel_db.h"

/**
 * CPUCache - Create a per thread buffer that can fit in L1/L2 cache 
 *            and is aligned to the page size
 */
struct CPUCache {
private:
  /**
   * @threads: Number of threads.
   */
  uint32_t threads;
  /**
   * @size: Size of each buffer.
   */
  uint32_t size;

public:
  /**
   * @ptr: Pointer to buffer for each thread.
   */
  void** ptr;

  CPUCache() : threads(0), size(0), ptr(nullptr) {}

  void alloc(uint32_t threads, uint32_t size) {
    this->threads = threads;
    uint32_t pageSize = getpagesize();
    this->size = (size/pageSize + 1) * pageSize;
    ptr = (void**)malloc(threads * sizeof(void*));
    for (uint32_t i = 0; i < threads; i++) {
      ptr[i] = aligned_alloc(pageSize, this->size);
    }
  }

  ~CPUCache() {
    for (uint32_t i = 0; i < threads; i++) {
      free(ptr[i]);
    }

    free(ptr);
    ptr = nullptr;
    threads = 0;
    size = 0;
  }
};

/**
 * CPUKernelDatabase executes CPU Kernels and is a subclass of KernelDatabase.
 */
class CPUKernelDatabase : public KernelDatabase {
protected:
  /**
   * @trash1, @trash2: Temporary buffers for clearing L1, L2, and L3 cache.
   */
  char* trash1, *trash2;
  /**
   * @TileXs, @TileYs, and @TileFs: Temporary cache buffers for X, Y, and F.
   */
  CPUCache TileXs;
  CPUCache TileYs;
  CPUCache TileFs;

public:
  CPUKernelDatabase();
  ~CPUKernelDatabase() {
    delete[] trash1; delete[] trash2;
    for (auto detail : hardware) delete detail;
  }

protected:
  /**
   * allocate_caches() - Allocate TileXs, TileYs, and TileFs.
   */
  void allocate_caches();

  /**
   * getMaxThreads() - Return maximum number of threads can execute in parallel
   */
  uint32_t getMaxThreads() const {
    return MAX((uint32_t)omp_get_max_threads(),
               getCPUProperties().sockets * getCPUProperties().cores);
  }

  /**
   * getCPUProperties() - Obtain hardware properties of the CPU
   */
  CPUArchDetails getCPUProperties() const {
    return *(dynamic_cast<const CPUArchDetails*>(hardware[0]));
  }

public:
  /**
   * procMemset() - Overriding KernelDatabase::procMemset.
   */
  virtual fastKronError procMemset(uint32_t proc, Matrix& m, float val);
  
protected:
  /**
   * procMalloc() - Overriding KernelDatabase::procMalloc.
   */
  virtual fastKronError procMalloc(uint32_t proc, size_t size, void*& ptr);
  /**
   * procFree() - Overriding KernelDatabase::procFree.
   */
  virtual fastKronError procFree(uint32_t proc, void* ptr);

public:
  /**
   * initTune() - Overriding KernelDatabase::initTune.
   */
  virtual fastKronError initTune() {return fastKronSuccess;}
  
  /**
   * invokeKernel() - Overriding KernelDatabase::invokeKernel
   */
  virtual fastKronError invokeKernel(KMMKernel* kernel, KMMProblem problem,
                                     const uint fidx,
                                     EpilogueParams epilogueParams,
                                     KernelMode execMode);
  virtual fastKronError invokeKernel(KMMKernel* kernel, KMMProblemStridedBatched problem,
                                     const uint fidx,
                                     EpilogueStridedBatchedParams epilogueParams,
                                     KernelMode execMode);
  /**
   * invokeP2PStoreKernel() - Overriding KernelDatabase::invokeP2PStoreKernel
   * FUTURE WORK: Do  not support multi node GEKMM on CPUs
   */
  virtual fastKronError invokeP2PStoreKernel(KMMKernel*, KMMProblem,
                                             const uint,
                                             DistributedParams, 
                                             EpilogueParams,
                                             KernelMode) {return fastKronSuccess;}
private:
  template<typename KMMProblem, typename EpilogueParams>
  fastKronError invokeKernel(KMMKernel* kernel, KMMProblem problem,
                                     const uint fidx,
                                     EpilogueParams epilogueParams,
                                     KernelMode execMode);
public:
  /**
   * timeKernel() - Overriding KernelDatabase::timeKernel
   */

  virtual fastKronError timeKernel(KMMKernel* kernel, KMMProblem problem, 
                                   const uint fidx, 
                                   DistributedParams distParams,
                                   EpilogueParams epilogueParams,
                                   KernelMode execMode, 
                                   bool useP2PStore,
                                   int warmups, int runs,
                                   float& runtime);
  
  virtual fastKronError timeKernel(KMMKernel* kernel, KMMProblemStridedBatched problem, 
                                   const uint fidx, 
                                   DistributedParams distParams,
                                   EpilogueStridedBatchedParams epilogueParams,
                                   KernelMode execMode, 
                                   bool useP2PStore,
                                   int warmups, int runs,
                                   float& runtime);
private:
  template<typename KMMProblemT, typename EpilogueParamsT>
  fastKronError timeKernel(KMMKernel* kernel, KMMProblemT problem, 
                           const uint fidx, 
                           DistributedParams distParams,
                           EpilogueParamsT epilogueParams,
                           KernelMode execMode, 
                           bool useP2PStore,
                           int warmups, int runs,
                           float& runtime);
protected:
  /**
   * occupancyDetails() - Overriding KernelDatabase::occupancyDetails.
   */ 
  virtual std::string occupancyDetails(KMMKernel*, KMMProblem) {return "";}
  virtual std::string occupancyDetails(KMMKernel*, KMMProblemStridedBatched) {return "";}
};

/**
 * X86KernelDatabase - executes X86 Kernels and is a subclass of CPUKernelDatabase.
 */
class X86KernelDatabase : public CPUKernelDatabase {
public:
  X86KernelDatabase();

private:
  /**
   * getX86CPUProperties() - Obtain properties of underlying x86 CPU.
   */
  X86ArchDetails getX86CPUProperties() const {return *(dynamic_cast<const X86ArchDetails*>(hardware[0]));}

protected:
  /**
   * findKernelAtOptLevel() - Overriding KernelDatabase::findKernelAtOptLevel
   */
  virtual KMMKernel* findKernelAtOptLevel(KMMProblem subProblem,
                                           const std::vector<KMMKernel*>& kernels) {
    return findKernelAtOptLevel<KMMProblem>(subProblem, kernels);
  }
  virtual KMMKernel* findKernelAtOptLevel(KMMProblemStridedBatched subProblem, 
                                          const std::vector<KMMKernel*>& kernels) {
    return findKernelAtOptLevel<KMMProblemStridedBatched>(subProblem, kernels);
  }
  /**
   * findKernelAtOptLevel() - The template implementation of both findKernelAtOptLevel 
   *                          virtual methods.
   */
  template<typename KMMProblemT>
  KMMKernel* findKernelAtOptLevel(KMMProblemT subProblem,
                                  const std::vector<KMMKernel*>& kernels);
};