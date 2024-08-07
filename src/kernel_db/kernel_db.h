#include <functional>
#include <map>
#include <algorithm>

#include "kmm/kmmalgo.h"
#include "kernels/kernel_info.h"
#include "kernels/params.h"

#pragma once

template<>
struct std::hash<std::pair<Factor, uint32_t>> {
  std::size_t operator()(const std::pair<Factor, uint32_t>& m) const;
};

class OptimizedKernelForShape;

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

public:
  std::unordered_map<DbKey, std::vector<KernelInfo*>, DbKeyHash> compiledKernels;
  std::vector<KernelInfo*> allKernels;
  std::vector<HardwareDetails*> hardware;
  std::unordered_map<KMMProblem, TunedKernelsSeries> problemToKernelCache;

public:
  KernelDatabase();
  ~KernelDatabase() {}

  template<typename SubClassKernel>
  void loadKernels(SubClassKernel* kernels, uint32_t num) {
    //Load kernels into compiledKernels map
    for (uint i = 0; i < num; i++) {
      SubClassKernel& info = kernels[i];
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
  
  virtual bool findAllKernels(KMMProblem problem, bool distP2PStore, 
                              std::vector<std::vector<KernelInfo*>>& kernels) {
    for (uint32_t i = 0; i <= KernelOptimizations::MaxOptLevel(); i++) {
      kernels.push_back(std::vector<KernelInfo*>());
    }

    DbKey key = DbKey{problem.f(0), problem.opX(), problem.opFs()};
    auto it = compiledKernels.find(key);
    if (it != compiledKernels.end()) {
      for (auto k : it->second) {
        if (k->canCompute(problem, hardware[0], distP2PStore) &&
            k->OptLevel == KernelOptimizations::MaxOptLevel()) {
          kernels[k->OptLevel].push_back(k);
        }
      }
    }
  
    if (it != compiledKernels.end() and 
        kernels[KernelOptimizations::MaxOptLevel()].size() > 0)
        return true;

    for (auto it : compiledKernels) {
      for (auto kernel : it.second) {
        if (kernel->canCompute(problem, hardware[0], distP2PStore)) {
          kernels[kernel->OptLevel].push_back(kernel);
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
    [distP2PStore, problem, this](auto& kernel){return kernel->FusedFacs <= problem.n() && 
                                          kernel->OptLevel == KernelOptimizations::MaxOptLevel() &&
                                          kernel->canCompute(problem, this->hardware[0], distP2PStore, false);});
    return true;
  }


  std::pair<KernelInfo*, float> tuneKernelForProblem(KMMProblem problem, bool distP2PStore, uint factorIdx, DistributedParams distParams);
  TunedKernelsSeries kernelSeriesForProblem(KMMProblem problem);
  TunedKernelsSeries __kernelSeriesForProblem(KMMProblem problem);
  virtual std::map<uint32_t, std::vector<KernelInfo*>, std::greater<int>> filterFastestFusedKernels(const KMMProblem& problem, const std::vector<KernelInfo*>& kernels);
  virtual KernelInfo* kernelForSubProblem(KMMProblem subProblem, const std::vector<KernelInfo*>& kernels) = 0;
  KernelInfo* kernelForSubProblem(KMMProblem subProblem, const std::vector<std::vector<KernelInfo*>>& kernels) {
    for (int optlevel = KernelOptimizations::MaxOptLevel();
        optlevel >= 0; optlevel--) {
      std::vector<KernelInfo*> kernelsForOptLevel = kernels[optlevel];
      if (kernelsForOptLevel.size() > 0) {
        KernelInfo* info = kernelForSubProblem(subProblem, kernelsForOptLevel);
        if (info) return info;
      }
    }

    return nullptr;
  }

  void free() {
    compiledKernels.clear();
  }
};

//TODO: Needs to be removed
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
