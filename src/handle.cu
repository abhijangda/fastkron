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

#include "utils.h"
#include "handle.h"
#include "thread_pool.h"
#include "device/otherkernels.cuh"
#include "env.h"
#include "autotuner.h"
#include "kernel_defs.cuh"

/*TODOs:
 1. Using fusion or not should be an environemnt flag
 2. Debug message environment flag*/

std::size_t std::hash<KronMatmulShape>::operator()(const KronMatmulShape& k) const {
  return hash<uint>()(k.KronCols) ^ hash<uint>()(k.KronRows) ^ hash<uint>()(k.ColsA);
}

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
    const uint NumFusedKerns = min(LocalKrons, NumKronMats - i);
    uint currTempN = prevTempN;
    // printf("243: NumFusedKerns %d kronMat \n", NumFusedKerns);
    uint FusedKronMatCols[NumFusedKerns];
    uint FusedKronMatRows[NumFusedKerns];
    for (int k = 0; k < NumFusedKerns; k++) {
      FusedKronMatCols[k] = KronMatCols[kronMat - k];
      FusedKronMatRows[k] = KronMatRows[kronMat - k];
      currTempN = (currTempN/FusedKronMatRows[k])*FusedKronMatCols[k];
    }
  
    if (currTempN % gpusInK != 0) return false;
    prevTempN = currTempN;
  }

  return true;
}

KronMatmulShape FastKronHandle::maxCompiledColsA(KronMatmulShape shape) {
  while (compiledKernels.find(shape) == compiledKernels.end()) {
    shape.ColsA /= 2;
    if (shape.ColsA == 1) {
     break;
    }
  }

  return shape;
}

uint FastKronHandle::maxFusedKernels(KronMatmulShape shape) {
  uint numFusedKernels = 0;
  //Go through fused kernels starting from 1 
  //find if the shape exists for the fused kernel
  //if it exist then go to next fused kernel
  while (true) {
    shape.NumFusedKerns = numFusedKernels + 1;
    auto shapeFound = maxCompiledColsA(shape);
    if (shapeFound.ColsA == 1) {
      break;
    }
    numFusedKernels++;
  }

  return numFusedKernels;
}

KernelInfo FastKronHandle::selectKernel(KronMatmulShape shape) {
  //Go through all MaxColsA starting from MAX_K and select the relevant
  KronMatmulShape maxColsAShape = maxCompiledColsA(shape);
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
      uint tileRowA = info.TileRowsA;
      bool row_mod_tile_zero = (shape.RowsA % tileRowA) == 0;    
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
  uint MaxFusedKerns = getUseFusion() ? maxFusedKernels(KronMatmulShape{KronMatCols[0], KronMatRows[0], K, M, 0}) : 1;
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
    auto selectedKernel = selectKernel(KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                       prevTempN, M, NumFusedKerns, DistributeToGPUs});
    tunedSeries.push_back({selectedKernel, kronMat - NumFusedKerns, kronMat, prevTempN, 0.0f});
    prevTempN = currTempN;
  }

  return tunedSeries;
}

cudaError_t FastKronHandle::xgekmm(const uint NumKronMats, void* x, void** kronMats,
                                void* result,
                                uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], 
                                void* temp1, void* temp2, 
                                EpilogueParams epilogueParams,
                                cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  if (result == nullptr) return cudaErrorInvalidValue;
  if (temp1  == nullptr) return cudaErrorInvalidValue;

  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return cudaErrorInvalidValue;
  
  void* kronGemmResults[2] = {temp1, temp2};
  void* prevKronResult = x;
  void* currKronResult = kronGemmResults[0];

  //TODO: Assumes all factors are of same size and square shape
  TunedKernelsSeries kernelSeries;
  if (tunedKernelSeries.size() > 0) {
    kernelSeries = tunedKernelSeries;
  } else {
    kernelSeries = selectKernelSeries(NumKronMats, M, N, K, 
                                      KronMatCols, KronMatRows, false);
  }

  if (temp2 == nullptr) {
    if (kernelSeries.size() % 2 == 1) {
      kronGemmResults[0] = result;
      kronGemmResults[1] = temp1;
    } else {
      kronGemmResults[0] = temp1;
      kronGemmResults[1] = result;
    }

    currKronResult = kronGemmResults[0];
    prevKronResult = x;
  }

  //Use double buffering for writing result and using output 
  //of previous iteration as input to current
  uint prevTempN = K;
  uint currTempN;
  for (auto kernel : kernelSeries) {
    const uint kronMat = kernel.end;
    const uint NumFusedKerns = kernel.kernel.NumFusedKerns;
    void* krons[NumFusedKerns];
    uint FusedKronMatCols[NumFusedKerns];
    uint FusedKronMatRows[NumFusedKerns];
    currTempN = prevTempN;
    for (int k = 0; k < NumFusedKerns; k++) {
      krons[k] = kronMats[kronMat - k];
      FusedKronMatCols[k] = KronMatCols[kronMat - k];
      FusedKronMatRows[k] = KronMatRows[kronMat - k];
      currTempN = (currTempN/FusedKronMatRows[k])*FusedKronMatCols[k];
    }

    //In the last iteration, write result to the results.    
    if (kronMat - NumFusedKerns + 1 == 0)
      currKronResult = result;

    cudaError_t status;

    KernelInfo selectedKernel = kernel.kernel;
    // std::cout << "Invoking " << selectedKernel << " for " << FusedKronMatCols[0] << "x" << FusedKronMatRows[0] << "  " << prevTempN << " " << currTempN << std::endl;
    status = kernelInvoker.fusedSlicedMatmul(NumFusedKerns, selectedKernel, kronMat, (void*)prevKronResult,
                               (void**)krons, (void*)currKronResult, M, currTempN, prevTempN,
                               FusedKronMatCols, FusedKronMatRows,
                               epilogueParams, stream);
    
    if (status != cudaSuccess) return status;
    
    // if (kronMat >= 1)
    // printGPUArray<float>(M, currTempN, (kronMat == 3) ? 8.0f : (kronMat == 2 ? 64.0f : 512.0f),
    //                      (float*)currKronResult, stream);
    // if (kronMat == 3) return cudaSuccess;
    prevTempN = currTempN;
    // if (kronMat == 1) return cudaSuccess;
    // return cudaSuccess;
    //Double/ring/circular buffer previous result and new result
    prevKronResult = currKronResult;
    if (prevKronResult == kronGemmResults[0]) {        
      currKronResult = kronGemmResults[1];
    } else if (prevKronResult == kronGemmResults[1]) {
      currKronResult = kronGemmResults[0];
    }
  }

  return cudaSuccess;
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
    // gpuM_ = M_/gpusInM_;
    // gpuK_ = K_/gpusInK_;
    // gpuN_ = N_/gpusInK_;
    
    //All gpus with same row shares the same barrier
    //TODO: free
    barriers_ = new pthread_barrier_t[gpusInM_];
    threads_ = new thread_pool<ThreadArgs*>(numGPUs_);

    for (int i = 0; i < gpusInM_; i++) {
      int s = pthread_barrier_init(&barriers_[i], NULL, gpusInK_);
      PTHREAD_BARRIER_CHECK(s);
    }
    
    // size_t tempN = gpuK_;
    // size_t maxTempN = tempN;
    // for (int i = 0; i < NumKronMats_; i++) {
    //   tempN = (tempN/KronMatRows_[i])*KronMatCols_[i];
    //   if (maxTempN < tempN)
    //     maxTempN = tempN;
    // }

    // size_t sz = gpuM_ * maxTempN * sizeof(T);
    // std::cout << "Allocating temporaries of size "<< sz << std::endl;
    // std::cout << "Allocated temporaries"<<std::endl;

  }

  //Load kernels into compiledKernels map
  for (uint i = 0; i < sizeof(KronGemmKernels)/sizeof(KernelInfo); i++) {
    KernelInfo& info = KronGemmKernels[i];
    KronMatmulShape shape {info.KronCols, info.KronRows, info.MaxColsA, 0, info.NumFusedKerns, info.DistributeToGPUs};
    auto iter = compiledKernels.find(shape);
    if (iter == compiledKernels.end()) {
      compiledKernels.emplace(std::make_pair(shape, std::vector<KernelInfo>()));
    }
    compiledKernels.at(shape).push_back(info);
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