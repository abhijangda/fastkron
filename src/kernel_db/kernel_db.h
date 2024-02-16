#include <functional>

#include <cuda_runtime.h>
#include <cuda.h>

#include "kmm/kmmalgo.h"
#include "kernels/kernel_info.h"
#include "kernels/params.h"

class KernelDatabase {
public:
  struct DbKey {
    Factor f;
    fastKronOp opX;
    fastKronOp opF;

    bool operator==(const DbKey& other) const {
      return other.f == f && other.opX == opX && other.opF == opF;
    }
  };

  struct DbKeyHash {
    size_t operator()(const DbKey& k) const {
      return std::hash<Factor>()(k.f) ^ std::hash<uint32_t>()(k.opX) ^ std::hash<uint32_t>()(k.opF);
    }
  };

protected:
  std::unordered_map<DbKey, std::vector<KernelInfo>, DbKeyHash> compiledKernels;

public:
  KernelDatabase() {}

  void free() {
    compiledKernels.clear();
  }
  
  virtual cudaError_t invokeKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                   KMMProblem problem,
                                   EpilogueParams epilogueParams,
                                   KernelMode execMode) = 0;
  virtual cudaError_t invokeP2PStoreKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                           KMMProblem problem, DistributedParams distParams, 
                                           EpilogueParams epilogueParams,
                                           KernelMode execMode) = 0;

  bool findAllKernels(const Factor& f, fastKronOp opX, fastKronOp opF, std::vector<KernelInfo>& kernels) {
    auto it = compiledKernels.find(DbKey{f, opX, opF});
    if (it == compiledKernels.end()) return false;
    kernels = it->second;
    return true;
  }

  virtual std::pair<KernelInfo, float> tuneKernelForProblem(KMMProblem problem, bool distP2PStore, uint factorIdx, DistributedParams distParams) = 0;
  virtual cudaError_t procMalloc(uint32_t proc, size_t size, void*& ptr) = 0;
  virtual cudaError_t procMalloc(uint32_t proc, Matrix& m) = 0;
  virtual cudaError_t procMemset(uint32_t proc, Matrix& m, float val) = 0;
  virtual cudaError_t procFree(uint32_t proc, Matrix m) = 0;
  virtual cudaError_t procFree(uint32_t proc, void* ptr) = 0;
};