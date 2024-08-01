#include <iostream>
#include <iomanip>
#include <vector>

#include "kernel_db/kernel_db.h"
#include "utils/logger.h"

KernelDatabase::KernelDatabase() {}

std::map<uint32_t, std::vector<KernelInfo*>, std::greater<int>> KernelDatabase::filterFastestFusedKernels(const KMMProblem& problem, const std::vector<KernelInfo*>& kernels) {
  std::map<uint32_t, std::vector<KernelInfo*>, std::greater<int>> numFusedToKernels;

  for (auto kernel : kernels) {
    if (kernel->FusedFacs <= problem.n()) {
      if (numFusedToKernels.find(kernel->FusedFacs) == numFusedToKernels.end())
        numFusedToKernels[kernel->FusedFacs] = std::vector<KernelInfo*>();
      numFusedToKernels[kernel->FusedFacs].push_back(kernel);
    }      
  }

  if (false) {
    std::cout << "346: " << numFusedToKernels.size() << std::endl;
    for (auto iter : numFusedToKernels) {
      std::cout << iter.first << ": ";
      for (auto k : iter.second) {
        std::cout << k->str() << " ";
      }
      std::cout << std::endl;
    }
  }

  return numFusedToKernels;
}

TunedKernelsSeries KernelDatabase::__kernelSeriesForProblem(KMMProblem problem) {
  TunedKernelsSeries kernelSeries;
  uint32_t MaxFuseP = 32;

  //TODO: fusion should considered for subproblems
  bool factorsSameShape = true, factorsSquare = true, 
       factorsPowerOfTwoShape = true, factorsLessThanMaxP = true;
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
    std::vector<KernelInfo*> kernels;
    findAllFusedKernels(problem, false, kernels);
    //Use a kernel that processes full problem
    for (auto kernel : kernels) {
      if (problem.n() == kernel->FusedFacs) {
        kernelSeries.push_back(TunedKernelFromStart(kernel, 0, kernel->FusedFacs-1, problem.k(), 0.0f));
        goto end;
      }
    }
    bool firstOpTKernelFound = false;
    if (problem.opX() == fastKronOp_T) {
      //First kernel for OpX = T
      auto numFusedToKernels_T = filterFastestFusedKernels(problem, kernels);
      if (!numFusedToKernels_T.empty()) {
        auto maxFused = numFusedToKernels_T.begin();
        std::vector<std::vector<KernelInfo*>> k;
        for (uint32_t i = 0; i <= KernelOptimizations::MaxOptLevel(); i++)
          if (i == KernelOptimizations::MaxOptLevel())  
            k.push_back(maxFused->second);
          else
            k.push_back(std::vector<KernelInfo*>());

        auto tk = TunedKernelFromStart(this->kernelForSubProblem(problem, k), 
                                      problem.n() - maxFused->first, problem.n() - 1, problem.k(), 0.0f);
        kernelSeries.push_back(tk);
        firstOpTKernelFound = true;

        //Find kernels for OpX = N
        KMMProblem subProblem = problem.rsub(problem.n() - 1 - maxFused->first, problem.n() - maxFused->first);
        subProblem.setOpX(fastKronOp_N);
        std::vector<KernelInfo*> kernels_OpN;
        findAllFusedKernels(subProblem, false, kernels_OpN);
        kernels = kernels_OpN;
        problem = subProblem;
      }
    } else {
      firstOpTKernelFound = true;
    }
    
    auto numFusedToKernels = filterFastestFusedKernels(problem, kernels);

    if (firstOpTKernelFound && !numFusedToKernels.empty()) {
      auto fusedIter = numFusedToKernels.begin();

      executeGeKMM(problem, nullptr, problem.n(),
        [&fusedIter](const KMMProblem) {return fusedIter->first;},
        [&fusedIter, &kernelSeries, &numFusedToKernels, this]
          (const KMMProblem subProblem, int rstart, void*[2], Matrix) {
            auto tk = TunedKernelFromStart(this->kernelForSubProblem(subProblem, fusedIter->second), 
                                          rstart - (subProblem.n() - 1), rstart, subProblem.k(), 0.0f);
            kernelSeries.push_back(tk);
            // std::cout << "370 " << fusedIter->first << " " << rstart << "  " << (rstart - subProblem.n() + 1) << std::endl; 
            while (fusedIter->first > rstart - subProblem.n() + 1&& fusedIter != numFusedToKernels.end()) {
                        // std::cout << "372 " << fusedIter->first << " " << rstart << "  " << (rstart - (subProblem.n() - 1)) << std::endl;
              fusedIter++;
            }
            return fastKronSuccess;
        });
    }

    if (!kernelSeries.empty()) goto end;
  }

  //No Fused kernel case found
  {
    executeGeKMM(problem, nullptr, problem.n(),
      [](const KMMProblem) {return 1;},
      [&kernelSeries, this]
        (const KMMProblem subProblem, int rstart, void*[2], Matrix) {
          std::vector<std::vector<KernelInfo*>> kernels;
          findAllKernels(subProblem, false, kernels);
          auto tk = TunedKernelFromStart(this->kernelForSubProblem(subProblem, kernels), 
                                         rstart, rstart, subProblem.k(), 0.0f);
          kernelSeries.push_back(tk);
          return fastKronSuccess;
      });
  }

end:
  Logger(LogLevel::Debug) << "Minimum Time " << std::endl;
  for (auto iter = kernelSeries.rbegin(); iter != kernelSeries.rend(); iter++) {
    Logger(LogLevel::Debug) << "  " << (*iter) << std::endl;
  }

  return kernelSeries;
}

TunedKernelsSeries KernelDatabase::kernelSeriesForProblem(KMMProblem problem) {
  if (problemToKernelCache.find(problem) != problemToKernelCache.end())
    return problemToKernelCache[problem];

  auto kernelSeries = __kernelSeriesForProblem(problem);

  problemToKernelCache[problem] = kernelSeries;

  return kernelSeries;
}

std::pair<KernelInfo*, float> KernelDatabase::tuneKernelForProblem(KMMProblem problem, bool distP2PStore, 
    uint factorIdx, DistributedParams distParams) {
  const uint runs = 5;
  const uint warmups = 5;
  KernelInfo* bestKernel = nullptr;
  float minTime;
  std::vector<std::vector<KernelInfo*>> allKernels;

  minTime = std::numeric_limits<float>::max();

  Logger(LogLevel::Debug) << "Tuning for shape "  << problem << std::endl;
  if (findAllKernels(problem, distP2PStore, allKernels)) {
  const std::vector<KernelInfo*>& kernelsForMaxOpt = [&allKernels]() {
    for (auto iter = allKernels.rbegin(); iter != allKernels.rend(); iter++) {
      if (iter->size() > 0) return *iter;
    }
    return allKernels[0];
  }();

  uint32_t kernelIdx = 0;
  for (auto kernel : kernelsForMaxOpt) {
    kernelIdx += 1;
    if (!kernel->canCompute(problem, hardware[0], distP2PStore)) continue;
    Logger(LogLevel::Debug) << "Kernel " << kernelIdx << "/" << kernelsForMaxOpt.size() << ": " << kernel->str() << std::endl;
    float kernelTime = std::numeric_limits<float>::max();
    fastKronError status;
    status = timeKernel(kernel, factorIdx, problem, distParams, EpilogueParams::create<float>(), KernelModeTuning, 
               distP2PStore, warmups, runs, kernelTime);
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
  }}

  if (minTime < std::numeric_limits<float>::max()) {
    Logger(LogLevel::Debug) << std::fixed << std::setprecision(4) <<
                "Fastest kernel for " << problem << ": " << bestKernel->str() << " runs in " << minTime << " ms" << std::endl;
    return std::make_pair(bestKernel, minTime);
  }

  return std::make_pair(bestKernel, minTime);
}

fastKronError KernelDatabase::procMalloc(uint32_t proc, FastKronType type, Matrix& m) {
  void* ptr = nullptr;
  fastKronError e = procMalloc(proc, m.numel() * sizeOfFastKronType(type), ptr);

  if (e == fastKronSuccess) {
    m.ptr = ptr;
  }

  return e;
}

fastKronError KernelDatabase::procFree(uint32_t proc, Matrix m) {
  return procFree(proc, m.data());
}

bool KernelInfo::validOptFor(KMMProblem problem, KernelOptimizations::Optimization opt) {
  using Opts = KernelOptimizations::Optimization;
  switch (opt) {
    case Opts::None:
      return true;
    case Opts::XshSlicesSame:
      return getTileX(problem).n()/problem.f(0).p() == tileX.n()/f.p();
    case Opts::QMultipleOfTileQ:
      return problem.f(0).q() % tileF.q() == 0;
    case Opts::PMultipleOfTileP:
      return problem.f(0).p() % tileF.p() == 0;
    case Opts::KMultipleOfTileK:
      return problem.k() % getTileX(problem).n() == 0;
    case Opts::QLeTileQ:
      return problem.f(0).q() <= f.q();
    case Opts::TileKSame:
      return getTileX(problem).n() == tileX.n();
    case Opts::FactorShapeSame:
      return f.p() == problem.f(0).p() && f.q() == problem.f(0).q();
    
    default:
      return false;
  }

  return false;
}