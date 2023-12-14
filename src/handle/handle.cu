#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <type_traits>
#include <thread>

#include <vector>
#include <iostream>
#include <sstream>
#include <limits>
#include <iomanip>
#include <cstring>
#include <functional>
#include <algorithm>

#include "utils/utils.h"
#include "utils/thread_pool.h"
#include "handle/handle.h"
#include "handle/kernel_defs.cuh"
#include "device/otherkernels.cuh"
#include "env/env.h"
#include "autotuner/autotuner.h"
#include "kmm/kmmalgo.h"

/*TODOs:
 1. Using fusion or not should be an environemnt flag
 2. Debug message environment flag*/

/**Library entry points to launch cuda kernels**/
//Check N and K is a multiplication of KronMatCols and KronMatRows
bool checkKronMatrixSizes(const uint NumKronMats, 
                          const uint M, const uint N, const uint K, 
                          const uint KronMatCols[], const uint KronMatRows[]) {
  uint n=1,k=1;
  for (uint i = 0; i < NumKronMats; i++) {
    k *= KronMatRows[i];
    n *= KronMatCols[i];
  }
  if (n != N || k != K) {
    printf("Invalid Kron product sizes %d != %d, %d != %d\n", n, N, k, K);
    return false;
  }

  return true;
}

bool checkDistributedKronSizes(const uint NumKronMats, 
                                      const uint M, const uint N, const uint K, 
                                      const uint KronMatCols[], const uint KronMatRows[],
                                      const uint LocalKrons, const uint gpusInK) {
  uint prevTempN = K;
  
  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return false;
  
  if (prevTempN % gpusInK != 0) return false;
    
  for (uint i = 0; i < NumKronMats; i += LocalKrons) {
    const uint kronMat = NumKronMats - i - 1;
    uint currTempN = prevTempN;
    for (int k = 0; k < min(LocalKrons, NumKronMats - i); k++) {
      currTempN = (currTempN/KronMatRows[kronMat - k])*KronMatCols[kronMat - k];
    }
  
    if (currTempN % gpusInK != 0) return false;
    prevTempN = currTempN;
  }

  return true;
}

bool checkDistributedKronSizes(const KMMProblem problem, const uint LocalN, const uint gpusInK) {
  bool correct = true;
  
  //Cannot do more than N local slicedmuls
  if (LocalN > problem.n) correct = false;

  //If Row is divided among then local slicedmuls has to be less than N 
  if (gpusInK > 1 and LocalN >= problem.n) correct = false;

  executeGeKMM(problem, nullptr, nullptr,
    [](const KMMProblem kmm) {return 1;},
    [&correct, gpusInK](const KMMProblem kmm, int, void* t1, void* t2) {
      correct = correct && (kmm.l % gpusInK == 0);
      return cudaSuccess;
    });
  return correct;
}

/*
SlicedMulShape FastKronHandle::maxCompiledColsA(SlicedMulShape shape) {
  while (compiledKernels.find(shape) == compiledKernels.end()) {
    shape.K /= 2;
    if (shape.K == 1) {
     break;
    }
  }

  return shape;
}

uint FastKronHandle::maxFusedKernels(SlicedMulShape shape) {
  uint numFusedKernels = 0;
  //Go through fused kernels starting from 1 
  //find if the shape exists for the fused kernel
  //if it exist then go to next fused kernel
  while (true) {
    shape.NumFusedKerns = numFusedKernels + 1;
    auto shapeFound = maxCompiledColsA(shape);
    if (shapeFound.K == 1) {
      break;
    }
    numFusedKernels++;
  }

  return numFusedKernels;
}

KernelInfo FastKronHandle::selectKernel(SlicedMulShape shape) {
  //Go through all MaxColsA starting from MAX_K and select the relevant
  SlicedMulShape maxColsAShape = maxCompiledColsA(shape);
  //TODO: Remove kEqVar. it provides only a little improvement in perf
  //but makes writing code hard
  int kEqVar = 0; //(maxColsAShape.ColsA == shape.ColsA) ? 1 : 0;
  auto iter = compiledKernels.find(maxColsAShape);
  if (iter == compiledKernels.end()) {
    std::cout << "No kernel found for " << shape << std::endl;
    abort();
    return KernelInfo{};
  }
  auto kernelInfos = iter->second;
  KernelInfo kernelInfo;
  for (auto info : kernelInfos) {
    //TODO: need to check for type
    //TODO: make use of KernelInfo.canCompute
    if (info.KEqVar == kEqVar) {
      bool row_mod_tile_zero = (shape.M % info.tiledShape.M) == 0;    
      if (info.RowModTileIsZero == row_mod_tile_zero) {
        return info;
      }
    }
  }

  std::cout<<"No kernel selected" << std::endl;
  abort();
  return KernelInfo();
}

//TODO: These methods that take handle should be private methods of FastKronHandle
TunedKernelsSeries FastKronHandle::selectKernelSeries(const uint NumKronMats,
                                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                                      bool distributedKernel) {
  uint MaxFusedKerns = getUseFusion() ? maxFusedKernels(SlicedMulShape{KronMatCols[0], KronMatRows[0], K, M, 0}) : 1;
  MaxFusedKerns = min(MaxFusedKerns, NumKronMats);
  TunedKernelsSeries tunedSeries;
  uint prevTempN = K;
  for (uint i = 0; i < NumKronMats; i += MaxFusedKerns) {
    const uint kronMat = NumKronMats - i - 1;
    const uint NumFusedKerns = min(MaxFusedKerns, NumKronMats - i);
    uint currTempN = prevTempN;
    // printf("243: NumFusedKerns %d kronMat \n", NumFusedKerns);
    uint FusedKronMatCols[NumFusedKerns];
    uint FusedKronMatRows[NumFusedKerns];
    for (int k = 0; k < NumFusedKerns; k++) {
      FusedKronMatCols[k] = KronMatCols[kronMat - k];
      FusedKronMatRows[k] = KronMatRows[kronMat - k];
      currTempN = (currTempN/FusedKronMatRows[k])*FusedKronMatCols[k];
    }
  
    bool DistributeToGPUs = distributedKernel && distComm_ == DistComm::P2P && gpusInK_ > 1 && (i == NumKronMats - 1);
    auto selectedKernel = selectKernel(SlicedMulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                       prevTempN, M, NumFusedKerns, DistributeToGPUs});
    tunedSeries.push_back({selectedKernel, kronMat - NumFusedKerns, kronMat, prevTempN, 0.0f});
    prevTempN = currTempN;
  }

  return tunedSeries;
}
*/

cudaError_t FastKronHandle::xgekmm(uint M, uint N, uint Ps[], uint Qs[], 
                                   void* X, void* Fs[], void* Y, void* temp1, void* temp2,
                                   EpilogueParams epilogueParams, cudaStream_t stream) {
  TunedKernelsSeries kernelSeries;
  if (tunedKernelSeries.size() > 0) {
    kernelSeries = tunedKernelSeries;
  } 
  // else {
  //   const uint K = std::reduce(Ps, Ps + N, 1, std::multiplies<uint>());
  //   const uint L = std::reduce(Qs, Qs + N, 1, std::multiplies<uint>());
  //   kernelSeries = selectKernelSeries(N, M, L, K, Qs, Ps, false);
  // }

  if (Y == nullptr) return cudaErrorInvalidValue;
  if (temp1 == nullptr) return cudaErrorInvalidValue;

  void* temps[2] = {temp1, temp2};
  void* input = X;
  void* output = temps[0];

  if (temp2 == nullptr) {
    if (kernelSeries.size() % 2 == 1) {
      temps[0] = Y;
      temps[1] = temp1;
    } else {
      temps[0] = temp1;
      temps[1] = Y;
    }

    output = temps[0];
    input = X;
  }

  KMMProblem problem(M, N, Ps, Qs, input, Fs, output);

  auto kernelSeriesIter = kernelSeries.begin();
  cudaError_t err = executeGeKMM(problem, temps, Y,
    [&kernelSeriesIter](const KMMProblem) {return kernelSeriesIter->kernel.NumFusedKerns_;},
    [&kernelSeriesIter, &err, epilogueParams, stream, this](const KMMProblem problem, int rstart, void* temps[2], void* result) {
      auto kernel = *kernelSeriesIter;
      
      KernelInfo selectedKernel = kernel.kernel;
      assert(rstart == kernel.end);
      err = this->kernelInvoker.invokeKernel(selectedKernel, rstart, 
                                             problem, epilogueParams,
                                             stream);
    
      CUDA_CHECK(err);
      kernelSeriesIter++;

      return err;
    });

  return err;
}

cudaError_t FastKronHandle::gekmmSizes(KMMProblem problem, size_t* resultSize, size_t* tempSize) {
  if (resultSize == nullptr) return cudaErrorInvalidValue;
  if (tempSize   == nullptr) return cudaErrorInvalidValue;

  uint gpuM, gpuK;

  if (isDistributed_) {
    if (!checkDistributedKronSizes(problem, perGPUKronBatch_, gpusInK_))
      return cudaErrorInvalidValue;
    gpuM = problem.m/gpusInM_;
    gpuK = problem.k/gpusInK_;
  } else {
    gpuM = problem.m;
    gpuK = problem.k;
  }

  int maxTempN = 0;
  int resultCols = 0;
                     
  auto e = executeGeKMM(problem, nullptr, nullptr,
    [](const KMMProblem kmm) {return 1;},
    [&maxTempN, &resultCols](const KMMProblem kmm, int rstart, void* temps[2], void* result) {
                            maxTempN = std::max(maxTempN, std::max(kmm.k, kmm.l));
                            resultCols = kmm.l;
                            return cudaSuccess;
                          });

  *tempSize   = gpuM * maxTempN;
  if (isDistributed_ and distComm_ == DistComm::NCCL)
    //Include size of send and recv buffers 
    *tempSize = (*tempSize) * 2;
  *resultSize = gpuM * resultCols;

  return e;
}

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

FastKronHandle::FastKronHandle(int gpus, int gpusInM, int gpusInK, int gpuKrons) : tunedKernelSeries() {
  //TODO: Support both modes. Single Process multi gpu and multi process multi gpu
  useFusion_ = true;
  isDistributed_ = gpus > 1;
  numGPUs_ = gpus;

  if (isDistributed_) {
    bool allP2PAccess = true;
    for (int g1 = 0; g1 < gpus; g1++) {
      for (int g2 = 0; g2 < gpus; g2++) {
        if (g1 == g2) continue;
        int p2pAccess = -1;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&p2pAccess, g1, g2));
        if (p2pAccess == 0) {allP2PAccess = false; break;}
        CUDA_CHECK(cudaSetDevice(g1));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(g2, 0));
      }
      if (!allP2PAccess) break;
    }

    distComm_ = env::getDistComm();

    if (distComm_ == DistComm::P2P) {
      if (!allP2PAccess) {
        std::cout << "P2P Access among GPUs not available using NCCL" << std::endl;
        distComm_ = DistComm::DistCommNone;
      }
    } else if (distComm_ == DistComm::NCCL) {
      int devs[gpus];
      distComm_ = DistComm::NCCL;
      ncclUniqueId ncclId;
      ncclGetUniqueId(&ncclId);
      std::cout << "Initializing NCCL"<<std::endl;
      for (int i = 0; i < gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        ncclComms.push_back(nullptr);
        devs[i] = i;
      }
      NCCLCHECK(ncclCommInitAll(&ncclComms[0], gpus, devs));
    }

    if (distComm_ == DistComm::DistCommNone) {
      if (allP2PAccess) {
        distComm_ = DistComm::P2P;
      } else {
        int devs[gpus];
        distComm_ = DistComm::NCCL;
        ncclUniqueId ncclId;
        ncclGetUniqueId(&ncclId);
        std::cout << "Initializing NCCL"<<std::endl;
        for (int i = 0; i < gpus; i++) {
          CUDA_CHECK(cudaSetDevice(i));
          ncclComms.push_back(nullptr);
          devs[i] = i;
        }
        NCCLCHECK(ncclCommInitAll(&ncclComms[0], gpus, devs));
      }
    }

    std::cout << "Using " << distComm_ << " for distributed comm" << std::endl;

    if (gpusInK >= 1)
      gpusInK_ = gpusInK;
    else
      gpusInK_ = 2;//ilog2(gpus);
    
    if (gpusInM >= 1)
      gpusInM_ = gpusInM;  
    else
      gpusInM_ = 1;//ilog2(gpus);
      
    //TODO: Check that gpuKrons batch is valid, i.e., P1*P2..PBatch <= gpusInK
    if (gpuKrons > 0)
      perGPUKronBatch_ = gpuKrons;
    else 
      perGPUKronBatch_ = 1;

    //TODO: Check if gpusInK_ == 1 then perGPUKronBatch = NumKrons

    std::cout << "gpusInRows " << gpusInM_ <<
                 " gpusInCols " << gpusInK_ << 
                 " gpuKronBatch " << perGPUKronBatch_ <<
                 std::endl;
    if (gpusInK_ * gpusInM_ != numGPUs_)  {
      std::cout << "gpusInCols * gpusInRows != total gpus (" << 
                   gpusInK_ * gpusInM_ << "!= " << 
                   numGPUs_<< ")" << std::endl;
      abort();
    }
    //TODO: Check that localKrons <= log (gpuK_)_P
    
    //All gpus with same row shares the same barrier
    //TODO: free
    barriers_ = new pthread_barrier_t[gpusInM_];
    threads_ = new thread_pool<ThreadArgs*>(numGPUs_);

    for (int i = 0; i < gpusInM_; i++) {
      int s = pthread_barrier_init(&barriers_[i], NULL, gpusInK_);
      PTHREAD_BARRIER_CHECK(s);
    }
  }

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

void FastKronHandle::free() {
  if (isDistributed_) {
    for (uint g = 0; g < gpusInM_; g++) {
      int s = pthread_barrier_destroy(&barriers_[g]);
      PTHREAD_BARRIER_CHECK(s);
    }

    delete threads_;
    delete barriers_;

    if (distComm_ == DistComm::NCCL) {
      for (int i=0; i<ncclComms.size(); i++)
        ncclCommDestroy(ncclComms[i]);
    }
  }
  compiledKernels.clear();
}

void FastKronHandle::getDistributedSizes(uint M, uint K, uint& gpuM, uint& gpuK) {
  gpuM = M/gpusInM_;
  gpuK = K/gpusInK_;
}