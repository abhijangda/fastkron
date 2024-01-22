#include <functional>

#include <cuda_runtime.h>
#include <cuda.h>

#include "kmm/kmmalgo.h"
#include "device/kernel_info.h"
#include "device/params.h"

#pragma once

//TODO: Change name to Executor?
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

private:
  std::unordered_map<DbKey, std::vector<KernelInfo>, DbKeyHash> compiledKernels;

public:
  KernelDatabase();
  void free() {
    compiledKernels.clear();
  }
  
  cudaError_t invokeKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                           KMMProblem problem,
                           EpilogueParams epilogueParams,
                           cudaStream_t stream);
  cudaError_t invokeP2PStoreKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                   KMMProblem problem, DistributedParams distParams, 
                                   EpilogueParams epilogueParams,
                                   cudaStream_t stream);
  bool findAllKernels(const Factor& f, fastKronOp opX, fastKronOp opF, std::vector<KernelInfo>& kernels) {
    auto it = compiledKernels.find(DbKey{f, opX, opF});
    if (it == compiledKernels.end()) return false;
    kernels = it->second;
    return true;
  }

  std::pair<KernelInfo, float> tuneKernelForProblem(KMMProblem problem, bool distP2PStore, uint factorIdx, DistributedParams distParams, cudaStream_t stream);
  cudaError_t procMalloc(uint32_t proc, size_t size, void*& ptr);
  cudaError_t procMalloc(uint32_t proc, Matrix& m);
  cudaError_t procFree(uint32_t proc, Matrix m);
  cudaError_t procFree(uint32_t proc, void* ptr); 

};