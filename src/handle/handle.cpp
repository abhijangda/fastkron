#include <cassert>
#include <iostream>

#include "utils/utils.h"
#include "utils/thread_pool.h"
#include "handle/handle.h"
#include "handle/op.h"
#include "env/env.h"
#include "autotuner/autotuner.h"
#include "kmm/kmmalgo.h"

/*TODOs:
 1. Using fusion or not should be an environemnt flag
 2. Debug message environment flag*/

/**Library entry points to launch cuda kernels**/

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

std::string fastKronOpToStr(const fastKronOp& op) {
  switch (op) {
    case fastKronOp_N:
      return "N";
    case fastKronOp_T:
      return "T";
  }

  return NULL;
}

std::ostream& operator<<(std::ostream& os, const fastKronOp& op) {
  os << fastKronOpToStr(op);
  return os;
}

cudaError_t FastKronHandle::xgekmm(const KMMProblem problem, void* temp1, void* temp2,
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
  if (problem.y().data() == nullptr) return cudaErrorInvalidValue;
  if (temp1           == nullptr) return cudaErrorInvalidValue;

  void* temps[2] = {temp1, temp2};

  auto kernelSeriesIter = kernelSeries.begin();
  cudaError_t err = executeGeKMM(problem, temps, kernelSeries.size(),
    [&kernelSeriesIter](const KMMProblem) {return kernelSeriesIter->kernel.NumFusedKerns_;},
    [&kernelSeriesIter, epilogueParams, stream, this]
      (const KMMProblem subProblem, int rstart, void* temps[2], Matrix result) {
        cudaError_t err;
        auto kernel = *kernelSeriesIter;

        KernelInfo selectedKernel = kernel.kernel;
        assert(rstart == kernel.end);
        printf("rstart %d\n", rstart);
        err = kerneldb.invokeKernel(selectedKernel, rstart, 
                                    subProblem, epilogueParams,
                                    stream);
        CUDA_CHECK(err);
        kernelSeriesIter++;
        CUDA_CHECK(cudaDeviceSynchronize());
        // if (rstart == 1) {
        //   float* tt = new float[8 * 16384];
        //   CUDA_CHECK(cudaMemcpy(tt, subProblem.y().data(), 8*16384*sizeof(float), cudaMemcpyDeviceToHost));
        //   for (int i = 0; i < 8; i++) {
        //     for (int j = 0; j < 16384; j++) {
        //       if (i % 2 == 0) if (tt[i * 16384 + j] != 0.0f) printf("tt[%d * 16384 + %d] %f\n", i, j, tt[i * 16384 + j]);
        //     }
        //   }
        // }
        return err;
    });

  return err;
}

cudaError_t FastKronHandle::gekmmResultTemp(KMMProblem problem, Matrix& result, Matrix& temp) {
  if (isDistributed_) {
    if (!checkDistributedKronSizes(problem, perGPUKronBatch_, gpusInK_))
      return cudaErrorInvalidValue;
  }

  uint32_t tempCols = 0;
  uint32_t resultCols = 0;
  auto e = executeGeKMM(problem, nullptr, 0,
    [](const KMMProblem kmm) {return 1;},
    [&tempCols, &resultCols]
    (const KMMProblem kmm, int rstart, void* temps[2], Matrix result) {
      tempCols = std::max(tempCols, std::max(kmm.k(), kmm.l()));
      resultCols = kmm.l();
      return cudaSuccess;
    });
  
  uint gpuM;

  if (isDistributed_) {
    getDistributedSizes(problem.m(), tempCols,   gpuM, tempCols);
    getDistributedSizes(problem.m(), resultCols, gpuM, resultCols);
  } else {
    gpuM = problem.m();
  }

  result = Matrix(gpuM, resultCols);
  temp = Matrix(gpuM, tempCols);
  return e;
}

cudaError_t FastKronHandle::gekmmSizes(KMMProblem problem, size_t* resultSize, size_t* tempSize) {
  if (resultSize == nullptr) return cudaErrorInvalidValue;
  if (tempSize   == nullptr) return cudaErrorInvalidValue;

  Matrix result, temp;

  cudaError_t e = gekmmResultTemp(problem, result, temp);

  if (e == cudaSuccess) {
    *tempSize   = temp.numel();
    if (isDistributed_ and distComm_ == DistComm::NCCL)
      //Include size of send and recv buffers 
      *tempSize = (*tempSize) * 2;
    *resultSize = result.numel();

    *tempSize   = *tempSize   * sizeof(float);
    *resultSize = *resultSize * sizeof(float);
  }

  return e;
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

  kerneldb.free();
}

void FastKronHandle::getDistributedSizes(uint M, uint K, uint& gpuM, uint& gpuK) {
  gpuM = M/gpusInM_;
  gpuK = K/gpusInK_;
}