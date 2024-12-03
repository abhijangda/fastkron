#include <iostream>
#include <iomanip>
#include <vector>

#include "kernel_db/kernel_db.h"
#include "utils/logger.h"

KernelDatabase::KernelDatabase() {}

fastKronError KernelDatabase::procMalloc(uint32_t proc, FastKronType type, Matrix& m) {
  void* ptr = nullptr;
  fastKronError e = procMalloc(proc, m.numel() * sizeOfFastKronType(type), ptr);

  if (e == fastKronSuccess) {
    m.ptr = ptr;
  }

  return e;
}

fastKronError KernelDatabase::procMalloc(uint32_t proc, FastKronType type,
                                         StridedBatchMatrix& m, int batches) {
  void* ptr = nullptr;
  fastKronError e = procMalloc(proc, m.numel() * sizeOfFastKronType(type) * batches, ptr);

  if (e == fastKronSuccess) {
    m.ptr = ptr;
  }

  return e;
}

fastKronError KernelDatabase::procMalloc(uint32_t proc, FastKronType type,
                                         StridedBatchFactor& m, int batches) {
  void* ptr = nullptr;
  fastKronError e = procMalloc(proc, m.numel() * sizeOfFastKronType(type) * batches, ptr);

  if (e == fastKronSuccess) {
    m.ptr = ptr;
  }

  return e;
}

fastKronError KernelDatabase::procMemset(uint32_t proc, StridedBatchMatrix& m, 
                                         int batches, float val) {
  for (int b = 0; b < batches; b++) {
    auto subM = m.batch<float>(b);
    auto e = procMemset(proc, subM, val);
    if (e != fastKronSuccess) return e;
  }
  return fastKronSuccess;
}

fastKronError KernelDatabase::procMemset(uint32_t proc, StridedBatchFactor& m, 
                                         int batches, float val) {
  for (int b = 0; b < batches; b++) {
    auto subM = m.batch<float>(b);
    auto e = procMemset(proc, subM, val);
    if (e != fastKronSuccess) return e;
  }

  return fastKronSuccess;
}

fastKronError KernelDatabase::procFree(uint32_t proc, Matrix m) {
  return procFree(proc, m.data());
}

template<typename KMMProblemT, typename EpilogueParams>
std::pair<KMMKernel*, float> KernelDatabase::findTunedKernel(KMMProblemT problem, KernelBatchType::Ty batchType,
                                                             bool useP2PStore, uint fidx, 
                                                             DistributedParams distParams) {
  KMMKernel* bestKernel = nullptr;
  //A map of opt level to kernels at the opt level
  std::vector<std::vector<KMMKernel*>> allKernels;
  float minTime;
  const uint runs = 5;
  const uint warmups = 5;
  
  minTime = std::numeric_limits<float>::max();

  Logger(LogLevel::Debug) << "Tuning for shape "  << problem << std::endl;

  //Find all kernels for each opt level that can compute the problem
  if (findAllKernels(problem, batchType, useP2PStore, allKernels)) {
    //Only execute kernels that are at the max opt level.
    const bool OnlyMaxOptKernels = true;
    uint32_t kernelIdx = 0;
    uint32_t totalKernels = 0;
    for (auto iter = allKernels.rbegin(); iter != allKernels.rend(); iter++) {
      totalKernels += iter->size();
      if (OnlyMaxOptKernels && iter->size() > 0) break;
    }

    //Go through each kernel at the max opt level
    for (auto iter = allKernels.rbegin(); iter != allKernels.rend(); iter++) {
      for (auto kernel : *iter) {
        kernelIdx += 1;
        if (!kernel->canCompute(problem, hardware[0], useP2PStore, batchType)) continue;
        Logger(LogLevel::Debug) << "Kernel " << kernelIdx << "/" << totalKernels
                                << ": " << kernel->str() << std::endl;
        float kernelTime = std::numeric_limits<float>::max();
        fastKronError status = fastKronSuccess;
        status = timeKernel(kernel, problem, fidx, distParams,
                            EpilogueParams::template create<float>(), KernelModeTuning, 
                            useP2PStore, warmups, runs, kernelTime);
        if (status == fastKronSuccess) {
          Logger(LogLevel::Debug) << 
                      "  Time(ms): " << std::fixed << std::setprecision(4) << kernelTime << std::endl <<
                      "  GFLOPs: " << (((double)problem.flop())/(kernelTime/1e3))/1e9 << std::endl <<
                      occupancyDetails(kernel, problem) << std::endl;
          if (kernelTime < minTime) {
            bestKernel = kernel;
            minTime = kernelTime;
          }
        }
      }

      if (OnlyMaxOptKernels && iter->size() > 0) break;
    }
  }

  if (minTime < std::numeric_limits<float>::max()) {
    Logger(LogLevel::Debug) << std::fixed << std::setprecision(4) <<
                "Fastest kernel for " << problem << ": " << bestKernel->str() <<
                " runs in " << minTime << " ms" << std::endl;
    return std::make_pair(bestKernel, minTime);
  }

  return std::make_pair(bestKernel, minTime);
}

std::pair<KMMKernel*, float> KernelDatabase::findTunedKernel(KMMProblem problem, bool useP2PStore, 
                                                             uint fidx, DistributedParams distParams) {
  return findTunedKernel<KMMProblem, EpilogueParams>(problem, KernelBatchType::Normal, useP2PStore, fidx, distParams);
}

std::pair<KMMKernel*, float> KernelDatabase::findTunedKernel(KMMProblemStridedBatched problem, bool useP2PStore, 
                                                              uint fidx, DistributedParams distParams) {
  return findTunedKernel<KMMProblemStridedBatched, EpilogueStridedBatchedParams>(problem, KernelBatchType::StridedBatched, useP2PStore, fidx, distParams);
}

TunedKernelsSeries KernelDatabase::kernelSeriesForProblem(KMMProblem problem) {
  return kernelSeriesForProblem<KMMProblem>(problem, problemToKernelCache);
}

TunedKernelsSeries KernelDatabase::kernelSeriesForProblem(KMMProblemStridedBatched problem) {
  return kernelSeriesForProblem<KMMProblemStridedBatched>(problem, stridedBatchedProblemToKernelCache);
}

template<typename KMMProblemT, typename KernelCache>
TunedKernelsSeries KernelDatabase::kernelSeriesForProblem(KMMProblemT problem, KernelCache& problemToKernelCache) {
  //If a kernel series for the problem is already found then return that
  if (problemToKernelCache.find(problem) != problemToKernelCache.end())
    return problemToKernelCache[problem];
  
  KMMProblemT origProblem = problem;
  typename KMMProblemT::Matrices emptyIntermediates;

  TunedKernelsSeries kernelSeries;
  {
    //Use a fast algorithm to search for a good kernel series
    uint32_t MaxFuseP = 32;

    //Check if the problem can be computed by fused kernels 
    bool factorsSameShape = true,       //Are all factors have same shape 
         factorsSquare = true,          //Are all factors square matrices
         factorsPowerOfTwoShape = true, //Are all factors dimensions power of 2
         factorsLessThanMaxP = true;    //Are all factors dimensions less than MaxFuseP
    for (uint32_t f = 0; f < problem.n(); f++) {
      const Factor& fac = problem.f(f);
      factorsLessThanMaxP = factorsLessThanMaxP && (fac.p() <= MaxFuseP);
      factorsSquare = factorsSquare && (fac.p() == fac.q());
      factorsPowerOfTwoShape = factorsPowerOfTwoShape && isPowerOf2(fac.p()) && isPowerOf2(fac.q());
      if (f > 0) {
        factorsSameShape = factorsSameShape && fac.p() == problem.f(f-1).p() && fac.q() == problem.f(f-1).q();
      }
    }

    bool canFuse = problem.n() > 1 && factorsSameShape && factorsSquare && factorsPowerOfTwoShape && factorsLessThanMaxP;

    if (canFuse) {
      std::vector<KMMKernel*> kernels;
      findAllFusedKernels(problem, false, kernels);

      //Find if a kernel can fuse all factors of the problem in
      //one invocation
      for (auto kernel : kernels) {
        if (problem.n() <= kernel->getFusedFacs()) {
          uint32_t start = 0;
          uint32_t end = problem.n() - 1;
          kernelSeries.push_back(TunedKernelFromStart(kernel, start, end, problem.k(), 0.0f));
          goto end;
        }
      }

      bool firstOpTKernelFound = false;
      uint32_t subProblemStart = 0; //Used only for KMM
      if (problem.opX() == fastKronOp_T) {
        //First fused kernel for OpX = T is the one that has maximum number of fusion
        auto numFusedToKernels_T = filterFastestFusedKernels(problem, kernels);
        if (!numFusedToKernels_T.empty()) {
          auto maxFused = numFusedToKernels_T.begin();
          std::vector<std::vector<KMMKernel*>> k;
          for (uint32_t i = 0; i <= KernelOptimizations::MaxOptLevel(); i++)
            if (i == KernelOptimizations::MaxOptLevel())  
              k.push_back(maxFused->second);
            else
              k.push_back(std::vector<KMMKernel*>());

          uint32_t start;
          uint32_t end;
          start = problem.n() - maxFused->first;
          end = problem.n() - 1;
          auto tk = TunedKernelFromStart(this->findKernelForSubProblem(problem, k), 
                                         start, end, problem.k(), 0.0f);
          kernelSeries.push_back(tk);
          firstOpTKernelFound = true;

          //Filter remaining kernels for OpX = N and remaining number of factors
          KMMProblemT subProblem = problem.rsub(problem.n() - 1 - maxFused->first, problem.n() - maxFused->first);
          subProblem.setOpX(fastKronOp_N);
          std::vector<KMMKernel*> kernels_OpN;
          findAllFusedKernels(subProblem, false, kernels_OpN);
          kernels = kernels_OpN;
          problem = subProblem;
        }
      } else {
        firstOpTKernelFound = true;
      }

      auto numFusedToKernels = filterFastestFusedKernels(problem, kernels);
      if (firstOpTKernelFound && !numFusedToKernels.empty()) {
        //From above filtered kernels find the kernel series using a greedy approach.
        //The approach always selects the kernel with the maximum number of fusion
        auto fusedIter = numFusedToKernels.begin();

        executeGeMM(problem, emptyIntermediates,
          [&fusedIter](const KMMProblemT) {
            return fusedIter->first;
          },
          [subProblemStart, problem, &fusedIter, &kernelSeries, &numFusedToKernels, this]
            (const KMMProblemT subProblem, int rstart, typename KMMProblemT::Matrices) {
              uint32_t kstart;
              uint32_t kend;
              uint32_t remainingLength;

              kstart = rstart - (subProblem.n() - 1);
              kend = rstart;
              remainingLength = kstart;

              auto tk = TunedKernelFromStart(this->findKernelAtOptLevel(subProblem, fusedIter->second), 
                                             subProblemStart + kstart, subProblemStart + kend, subProblem.k(), 0.0f);
              kernelSeries.push_back(tk);
              while (fusedIter->first > remainingLength &&
                     fusedIter != numFusedToKernels.end()) {
                fusedIter++;
              }
              return fastKronSuccess;
          });
      }
      if (!kernelSeries.empty()) goto end;
    }

    //No Fused kernel case found
    {
      executeGeMM(problem, emptyIntermediates,
        [](const KMMProblemT) {return 1;},
        [&kernelSeries, this]
          (const KMMProblemT subProblem, int rstart, typename KMMProblemT::Matrices) {
            std::vector<std::vector<KMMKernel*>> kernels;
            findAllKernels(subProblem, KernelBatchType::Normal, false, kernels);
            auto tk = TunedKernelFromStart(this->findKernelForSubProblem(subProblem, kernels), 
                                          rstart, rstart, subProblem.k(), 0.0f);
            kernelSeries.push_back(tk);
            return fastKronSuccess;
        });
    }

  end:
    Logger(LogLevel::Info) << "Minimum Time " << std::endl;
    for (auto iter = kernelSeries.rbegin(); iter != kernelSeries.rend(); iter++) {
      Logger(LogLevel::Info) << "  #"<< ((kernelSeries.rend() - iter) - 1) << ": " << (*iter) << std::endl;
    }
  }

  problemToKernelCache[origProblem] = kernelSeries;
  return kernelSeries;
}

bool KernelDatabase::findAllFusedKernels(KMMProblem problem, bool useP2PStore,
                                         std::vector<KMMKernel*>& kernels,
                                         KernelBatchType::Ty batchType) {
  DbKey key = DbKey{problem.f(0), problem.opX(), problem.opFs(), batchType};
  auto it = compiledKernels.find(key);
  if (it != compiledKernels.end()) {
  std::copy_if(it->second.begin(), it->second.end(), std::back_inserter(kernels), 
    [useP2PStore, problem, batchType, this](auto& kernel){
      return kernel->getOptLevel() == KernelOptimizations::MaxOptLevel() &&
             kernel->canCompute(problem, this->hardware[0], useP2PStore, batchType, false);
    });
  }

  if (batchType == KernelBatchType::Normal) {
    findAllFusedKernels(problem, useP2PStore, kernels, KernelBatchType::StridedBatched);
  }
  
  return true;
}

bool KernelDatabase::findAllFusedKernels(KMMProblemStridedBatched problem, bool useP2PStore, std::vector<KMMKernel*>& kernels) {
  return findAllFusedKernels(problem.batchProblem(0), useP2PStore, 
                             kernels, KernelBatchType::StridedBatched);
}

template<typename KMMProblem>
bool KernelDatabase::findAllKernels(KMMProblem problem, KernelBatchType::Ty batchType, bool useP2PStore, 
                                    std::vector<std::vector<KMMKernel*>>& kernels) {
  for (uint32_t i = 0; i <= KernelOptimizations::MaxOptLevel(); i++) {
    kernels.push_back(std::vector<KMMKernel*>());
  }

  bool AllOptKernels = false;
  if (!AllOptKernels) {
    DbKey key = DbKey{problem.f(0), problem.opX(), problem.opFs(), batchType};
    auto it = compiledKernels.find(key);
    if (it != compiledKernels.end()) {
      for (auto k : it->second) {
        if (k->canCompute(problem, hardware[0], useP2PStore, batchType) &&
            k->getOptLevel() == KernelOptimizations::MaxOptLevel()) {
          kernels[k->getOptLevel()].push_back(k);
        }
      }
    }

    if (it != compiledKernels.end() and 
        kernels[KernelOptimizations::MaxOptLevel()].size() > 0)
        return true;
  }

  for (auto it : compiledKernels) {
    for (auto kernel : it.second) {
      if (kernel->canCompute(problem, hardware[0], useP2PStore, batchType)) {
        kernels[kernel->getOptLevel()].push_back(kernel);
      }
    }
  }

  return true;
}

template<typename KMMProblemT>
KMMKernel* KernelDatabase::findKernelForSubProblem(KMMProblemT subProblem, 
                                                   const std::vector<std::vector<KMMKernel*>>& kernels) {
  //TODO: Only works for subproblem.n() == 1
  for (int optlevel = KernelOptimizations::MaxOptLevel();
       optlevel >= 0; optlevel--) {
    std::vector<KMMKernel*> kernelsForOptLevel = kernels[optlevel];
    if (kernelsForOptLevel.size() > 0) {
      KMMKernel* info = findKernelAtOptLevel(subProblem, kernelsForOptLevel);
      if (info) return info;
    }
  }

  return nullptr;
}

std::map<uint32_t, std::vector<KMMKernel*>, std::greater<int>> 
  KernelDatabase::filterFastestFusedKernels(const KMMProblem& problem, 
                                            const std::vector<KMMKernel*>& kernels) {
  std::map<uint32_t, std::vector<KMMKernel*>, std::greater<int>> numFusedToKernels;

  for (auto kernel : kernels) {
    //Always add a single fac kernel
    if (numFusedToKernels.find(1) == numFusedToKernels.end())
      numFusedToKernels[1] = {};
    numFusedToKernels[1].push_back(kernel);
    
    for (uint32_t numFusedFacs = 2; numFusedFacs <= kernel->getFusedFacs(); numFusedFacs++) {
      bool addFusedKernel = isFastFusedKernel(problem, kernel, numFusedFacs);
      if (addFusedKernel && numFusedFacs <= problem.n()) {
        if (numFusedToKernels.find(numFusedFacs) == numFusedToKernels.end())
          numFusedToKernels[numFusedFacs] = {};
        numFusedToKernels[numFusedFacs].push_back(kernel);
      }
    }
  }

  return numFusedToKernels;
}

KMMKernel* KernelDatabase::getKernel(std::string repr) {
  for (auto iter : compiledKernels) {
    for (auto kernel : iter.second) {
      if (kernel->str() == repr)
        return kernel;
    }
  }
  return nullptr;
}