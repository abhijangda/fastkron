#include <iostream>
#include <iomanip>
#include <vector>

#include "kernel_db/kernel_db.h"

std::pair<KernelInfo*, float> KernelDatabase::tuneKernelForProblem(KMMProblem problem, bool distP2PStore, 
    uint factorIdx, DistributedParams distParams) {
  const uint runs = 5;
  const uint warmups = 2;
  KernelInfo* bestKernel;
  float minTime;
  bool foundProblem = false;
  std::vector<KernelInfo*> allKernels;

  minTime = std::numeric_limits<float>::max();

  if (findAllKernels(problem.f(0), problem.opX(), problem.opFs(), allKernels)) {
  for (auto kernel : allKernels) {
    if (!kernel->canCompute(problem, distP2PStore)) continue;
    if (!foundProblem) {
      std::cout << "Tuning for shape "  << problem << std::endl;
      foundProblem = true;
    }
    std::cout << kernel->str();
    float kernelTime = std::numeric_limits<float>::max();
    cudaError_t status;
    status = timeKernel(kernel, factorIdx, problem, distParams, EpilogueParams::create<float>(), KernelModeTuning, 
               distP2PStore, 2, 5, kernelTime);
    if (status == cudaSuccess) {
      std::cout << std::fixed << std::setprecision(2) << 
                  " runs in " << kernelTime << " ms " << std::endl;
      if (kernelTime < minTime) {
        bestKernel = kernel;
        minTime = kernelTime;
      }
    }
  }}

  if (minTime < std::numeric_limits<float>::max()) {
    std::cout << std::fixed << std::setprecision(2) <<
                "Best kernel for " << problem << ": " << bestKernel << " runs in " << minTime << " ms" << std::endl;
    return std::make_pair(bestKernel, minTime);
  }

  return std::make_pair(bestKernel, minTime);
}

cudaError_t KernelDatabase::procMalloc(uint32_t proc, Matrix& m) {
  void* ptr = nullptr;
  cudaError_t e = procMalloc(proc, m.numel() * sizeof(float), ptr);

  if (e == cudaSuccess) {
    m.ptr = ptr;
  }

  return e;
}

cudaError_t KernelDatabase::procFree(uint32_t proc, Matrix m) {
  return procFree(proc, m.data());
}