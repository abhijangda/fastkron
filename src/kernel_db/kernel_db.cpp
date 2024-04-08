#include <iostream>
#include <iomanip>
#include <vector>

#include "kernel_db/kernel_db.h"

std::pair<KernelInfo*, float> KernelDatabase::tuneKernelForProblem(KMMProblem problem, bool distP2PStore, 
    uint factorIdx, DistributedParams distParams) {
  const uint runs = 5;
  const uint warmups = 5;
  KernelInfo* bestKernel;
  float minTime;
  std::vector<KernelInfo*> allKernels;

  minTime = std::numeric_limits<float>::max();

  std::cout << "Tuning for shape "  << problem << std::endl;
  if (findAllKernels(problem, distP2PStore, allKernels)) {
  for (auto kernel : allKernels) {
    if (!kernel->canCompute(problem, distP2PStore)) continue;
    std::cout << kernel->str() << std::endl;
    float kernelTime = std::numeric_limits<float>::max();
    fastKronError status;
    status = timeKernel(kernel, factorIdx, problem, distParams, EpilogueParams::create<float>(), KernelModeTuning, 
               distP2PStore, warmups, runs, kernelTime);
    if (status == fastKronSuccess) {
      std::cout << "  Time(ms): " << std::fixed << std::setprecision(4) << kernelTime << std::endl <<
                   occupancyDetails(kernel, problem) << std::endl;
      if (kernelTime < minTime) {
        bestKernel = kernel;
        minTime = kernelTime;
      }
    }
  }}

  if (minTime < std::numeric_limits<float>::max()) {
    std::cout << std::fixed << std::setprecision(4) <<
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
  }

  return false;
}