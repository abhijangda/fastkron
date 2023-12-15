#include <iostream>
#include <iomanip>

#include "utils/utils.h"

#include "kernel_db/kernel_db.h"
#include "kernel_db/kernel_defs.cuh"

static bool isValidKernel(KernelInfo& kernelInfo) {
  const uint NumThreads = kernelInfo.NumThreads;
  const uint CRegRows = kernelInfo.CRegRows;
  const uint CRegCols = kernelInfo.CRegCols;
  const Factor tiledFactor = kernelInfo.tiledFactor;

  const uint ValidThreads = ((kernelInfo.tiledInput.N/tiledFactor.P)/CRegRows) * (tiledFactor.Q/CRegCols);
  if (NumThreads != ROUNDUP(ValidThreads, CUDA_WARP_SIZE)) {
    std::cout << "Invalid kernel config " << kernelInfo << std::endl; 
    return false;
  }

  return true;
}

KernelDatabase::KernelDatabase() {
  //Load kernels into compiledKernels map
  for (uint i = 0; i < sizeof(KronGemmKernels)/sizeof(KernelInfo); i++) {
    KernelInfo& info = KronGemmKernels[i];
    if (!isValidKernel(info)) abort();
    //  {info.KronCols, info.KronRows, info.MaxColsA, 0, info.NumFusedKerns, info.DistributeToGPUs};
    auto iter = compiledKernels.find(info.factor);
    if (iter == compiledKernels.end()) {
      compiledKernels.emplace(std::make_pair(info.factor, std::vector<KernelInfo>()));
    }
    compiledKernels.at(info.factor).push_back(info);
  }
  
  //TODO: Check that if distP2PStore is needed then there is a kernel that can 
  //do it
  //TODO: Add if debug
  if (false) {
    uint numKernels = 0;
    std::cout << "Loading compiled kernels" << std::endl;
    for (auto iter : compiledKernels) {
      for (auto kernel : iter.second) {
        // std::cout << kernel << std::endl;
      }
      numKernels += iter.second.size();
    }
    std::cout << "Number of kernels loaded: " << numKernels << std::endl;
  }
}

//Launch cuda kernels
template<uint NumFusedKerns>
cudaError_t invoke(KernelInfo& kernelInfo, const uint kronIndex, 
                   KMMProblem problem,
                   DistributedParams distParams,
                   EpilogueParams epilogueParams,
                   cudaStream_t stream) {
  cudaError_t status;

  //Create the grid and thread block
  KernelParams<NumFusedKerns> params (problem, kronIndex);
  FusedParams<NumFusedKerns> fusedParams (problem, kernelInfo.tiledInput.N);

  //Call kernel
  typedef void (*KronMatmulKernelTy)(KernelParams<NumFusedKerns>, FusedParams<NumFusedKerns>, 
                                     DistributedParams, EpilogueParams, dim3, dim3, cudaStream_t);
  KronMatmulKernelTy(kernelInfo.kernel)(params, fusedParams, distParams, 
                                        epilogueParams, kernelInfo.grid(problem), 
                                        kernelInfo.block(), stream);
  status = cudaGetLastError();
  CUDA_CHECK(status);
  return status;
}

cudaError_t KernelDatabase::invokeKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                             KMMProblem problem, EpilogueParams epilogueParams,
                                             cudaStream_t stream) {
  DistributedParams distParams;

  switch(problem.n) {
    case 1:
      return invoke<1>(kernelInfo, kronIndex, problem,
                       distParams, epilogueParams, stream);
    case 2:
      return invoke<2>(kernelInfo, kronIndex, problem,
                       distParams, epilogueParams, stream);
    case 3:
      return invoke<3>(kernelInfo, kronIndex, problem,
                       distParams, epilogueParams, stream);
    case 4:
      return invoke<4>(kernelInfo, kronIndex, problem,
                       distParams, epilogueParams, stream);
    case 5:
      return invoke<5>(kernelInfo, kronIndex, problem,
                       distParams, epilogueParams, stream);
      break;
    default:
        std::cout << "Invalid number of fused kernels" << std::endl;
      return cudaErrorInvalidValue;
  }
}

cudaError_t KernelDatabase::invokeP2PStoreKernel(KernelInfo& kernel, const uint kronIndex, 
                                                KMMProblem problem, DistributedParams distParams, 
                                                EpilogueParams epilogueParams,
                                                cudaStream_t stream) {
  switch (problem.n) {
    case 1:
      return invoke<1>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, stream);
    case 2:
      return invoke<2>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, stream);
    case 3:
      return invoke<3>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, stream);
    case 4:
      return invoke<4>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, stream);
    case 5:
      return invoke<5>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, stream);
  }

  return cudaErrorInvalidValue;
}

std::pair<KernelInfo, float> KernelDatabase::tuneKernelForSize(KMMProblem problem, bool distP2PStore, uint factorIdx, DistributedParams distParams, cudaStream_t stream) {
  const uint runs = 5;
  const uint warmups = 2;
  KernelInfo bestKernel;
  cudaEvent_t start, end;
  float minTime;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  minTime = std::numeric_limits<float>::max();
  
  std::cout << "Tuning for shape "  << problem << std::endl;
  Factor factor(problem.qs[0], problem.ps[0]);

  for (auto shapeAndKernels : compiledKernels) {
    if (shapeAndKernels.first != factor) continue;
    for (auto kernel : shapeAndKernels.second) {
      if (!kernel.canCompute(problem, distP2PStore)) continue;
      CUDA_CHECK(cudaStreamSynchronize(stream));
      cudaError_t status;
      for (int r = 0; r < warmups + runs; r++) {
        if (r == warmups) CUDA_CHECK(cudaEventRecord(start, stream));
        if (distP2PStore) {
          status = invokeP2PStoreKernel(kernel, factorIdx, problem,
                                                distParams, EpilogueParams::create<float>(), stream);
        } else {
          status = invokeKernel(kernel, factorIdx, problem,
                                        EpilogueParams::create<float>(), stream);
        }
      }
      CUDA_CHECK(status);
      CUDA_CHECK(cudaEventRecord(end, stream));
      CUDA_CHECK(cudaEventSynchronize(end));
      
      if (status != cudaSuccess)
        std::cout << "Error: " << cudaGetErrorString(status) << " for " 
                  << kernel << " K " << problem.k << std::endl;
      float kernelTime;
      CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, end));
      std::cout << std::fixed << std::setprecision(2) << 
                  kernel << " runs in " << (kernelTime/runs) << " ms " << std::endl;
      if (kernelTime < minTime) {
        bestKernel = kernel;
        minTime = kernelTime;
      }
    }
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  if (minTime < std::numeric_limits<float>::max()) {
    std::cout << std::fixed << std::setprecision(2) <<
                "Best kernel for " << problem << ": " << bestKernel << " runs in " << (minTime/runs) << " ms" << std::endl;
    return std::make_pair(bestKernel, minTime/runs);
  }

  return std::make_pair(bestKernel, minTime);
}