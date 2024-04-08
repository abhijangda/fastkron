#include <functional>

#include "kmm/kmmalgo.h"
#include "kernels/kernel_info.h"
#include "kernels/params.h"

#pragma once

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

  bool isValidKernel(KernelInfo& info) {return true;} //TODO:
  void initKernel(KernelInfo& info) {return;} //TODO:
  
public:
  std::unordered_map<DbKey, std::vector<KernelInfo*>, DbKeyHash> compiledKernels;
  std::vector<KernelInfo*> allKernels;

public:
  KernelDatabase() {}
  ~KernelDatabase() {}

  template<typename SubClassKernel>
  void loadKernels(SubClassKernel* kernels, uint32_t num) {
    //Load kernels into compiledKernels map
    for (uint i = 0; i < num; i++) {
      SubClassKernel& info = kernels[i];
      if (!isValidKernel(info)) abort();
      initKernel(info); //CUDA_CHECK(info.setSharedMemAttr());
      //  {info.KronCols, info.KronRows, info.MaxColsA, 0, info.NumFusedKerns, info.DistributeToGPUs};
      DbKey key {info.f, info.opX, info.opF};
      auto iter = compiledKernels.find(key);
      if (iter == compiledKernels.end()) {
        compiledKernels.emplace(std::make_pair(key, std::vector<KernelInfo*>()));
      }
      compiledKernels.at(key).push_back(&info);
    }
  
    //TODO: Check that if distP2PStore is needed then there is a kernel that can 
    //do it
    //TODO: Add if debug
    if (false) {
      uint numKernels = 0;
      std::cout << "Loading compiled kernels" << std::endl;
      for (auto iter : compiledKernels) {
        for (auto kernel : iter.second) {
          std::cout << kernel->str() << std::endl;
        }
        numKernels += iter.second.size();
      }
      std::cout << "Number of kernels loaded: " << numKernels << std::endl;
    }
  }

  virtual fastKronError initTune() {return fastKronSuccess;}
  virtual fastKronError invokeKernel(KernelInfo* kernelInfo, const uint kronIndex, 
                                   KMMProblem problem,
                                   EpilogueParams epilogueParams,
                                   KernelMode execMode) = 0;
  virtual fastKronError invokeP2PStoreKernel(KernelInfo* kernelInfo, const uint kronIndex, 
                                           KMMProblem problem, DistributedParams distParams, 
                                           EpilogueParams epilogueParams,
                                           KernelMode execMode) = 0;
  virtual fastKronError timeKernel(KernelInfo* kernelInfo, const uint kronIndex, 
                                 KMMProblem problem, DistributedParams distParams, 
                                 EpilogueParams epilogueParams,
                                 KernelMode execMode, 
                                 bool distP2PStore,
                                 int warmups, int runs,
                                 float& runtime) = 0;
  virtual std::string   occupancyDetails(KernelInfo* kernelInfo, KMMProblem problem) = 0;
  virtual fastKronError procMalloc(uint32_t proc, size_t size, void*& ptr) = 0;
  virtual fastKronError procMemset(uint32_t proc, Matrix& m, float val) = 0;
  virtual fastKronError procFree(uint32_t proc, void* ptr) = 0;
  KernelInfo* getKernel(std::string repr) {
    for (auto iter : compiledKernels) {
      for (auto kernel : iter.second) {
        if (kernel->str() == repr)
          return kernel;
      }
    }
    return nullptr;
  }

  fastKronError procMalloc(uint32_t proc, FastKronType type, Matrix& m);
  fastKronError procFree(uint32_t proc, Matrix m);
  
  bool findAllKernels(KMMProblem problem, bool distP2PStore, std::vector<KernelInfo*>& kernels) {
    DbKey key = DbKey{problem.f(0), problem.opX(), problem.opFs()};
    auto it = compiledKernels.find(key);
    if (it != compiledKernels.end()) {
      for (auto k : it->second) {
        if (k->canCompute(problem, distP2PStore)) {
          kernels.push_back(k);
        }
      }
    }
    if (it != compiledKernels.end() and kernels.size() > 0) return true;

    for (auto it : compiledKernels) {
      if (it.first == key) continue;
      for (auto kernel : it.second) {
        if (kernel->canCompute(problem, distP2PStore)) {
          kernels.push_back(kernel);
        }
      }
    }

    return true;
  }

  bool findAllFusedKernels(KMMProblem problem, bool distP2PStore, std::vector<KernelInfo*>& kernels) {
    DbKey key = DbKey{problem.f(0), problem.opX(), problem.opFs()};
    auto it = compiledKernels.find(key);
    if (it == compiledKernels.end()) return false;
    std::copy_if(it->second.begin(), it->second.end(), std::back_inserter(kernels), 
    [distP2PStore, problem](auto& kernel){return kernel->FusedFacs <= problem.n() && 
                                          kernel->OptLevel == KernelOptimizations::MaxOptLevel() &&
                                          kernel->canCompute(problem, distP2PStore, false);});
    return true;
  }


  std::pair<KernelInfo*, float> tuneKernelForProblem(KMMProblem problem, bool distP2PStore, uint factorIdx, DistributedParams distParams);
  virtual TunedKernelsSeries kernelSeriesForProblem(KMMProblem problem) = 0;

  void free() {
    compiledKernels.clear();
  }
};