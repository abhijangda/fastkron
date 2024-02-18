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
                                   EpilogueParams epilogueParams) {
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
  if (temp1              == nullptr) return cudaErrorInvalidValue;

  void* temps[2] = {temp1, temp2};

  auto kernelDb = getKernelDb(backend);
  auto kernelSeriesIter = kernelSeries.begin();

  cudaError_t err = executeGeKMM(problem, temps, kernelSeries.size(),
    [&kernelSeriesIter](const KMMProblem) {return kernelSeriesIter->kernel.NumFusedKerns_;},
    [&kernelSeriesIter, epilogueParams, kernelDb, this]
      (const KMMProblem subProblem, int rstart, void* temps[2], Matrix result) {
        cudaError_t err;
        auto kernel = *kernelSeriesIter;

        KernelInfo selectedKernel = kernel.kernel;
        assert(rstart == kernel.end);
        err = kernelDb->invokeKernel(selectedKernel, rstart, 
                                     subProblem, epilogueParams,
                                     KernelModeNormal);
        CUDA_CHECK(err);
        kernelSeriesIter++;
        return err;
    });

  return err;
}

cudaError_t FastKronHandle::gekmmResultTemp(KMMProblem problem, Matrix& result, Matrix& temp) {
  if (cudaKernels.isDistributed_) {
    if (!checkDistributedKronSizes(problem, cudaKernels.perGPUKronBatch_, cudaKernels.gpusInK_))
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

  if (cudaKernels.isDistributed_) {
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
  //TODO: Should move to individual backend
  cudaError_t e = gekmmResultTemp(problem, result, temp);

  if (e == cudaSuccess) {
    *tempSize   = temp.numel();
    if (cudaKernels.isDistributed_ and cudaKernels.distComm_ == DistComm::NCCL)
      //Include size of send and recv buffers 
      *tempSize = (*tempSize) * 2;
    *resultSize = result.numel();

    *tempSize   = *tempSize   * sizeof(float);
    *resultSize = *resultSize * sizeof(float);
  }

  return e;
}

cudaError_t FastKronHandle::initCUDABackend(void* ptrToStream, int gpus, int gpusInM, int gpusInK, int gpuKrons) {
  if (backend != fastKronBackend_CUDA) return cudaErrorInvalidValue;
#ifdef ENABLE_CUDA
  cudaKernels.init((cudaStream_t*)ptrToStream, gpus, gpusInM, gpusInK, gpuKrons);
  return cudaSuccess;
#else
  return cudaErrorInvalidValue;
#endif
}

cudaError_t FastKronHandle::initX86Backend() {
  if (backend != fastKronBackend_X86) return cudaErrorInvalidValue;

  return cudaSuccess;
}

FastKronHandle::FastKronHandle(fastKronBackend backend) :
  tunedKernelSeries(), backend(backend), cudaKernels() {
  //TODO: Support both modes. Single Process multi gpu and multi process multi gpu
  useFusion_ = true;  
}

void FastKronHandle::free() {
  cudaKernels.free();
}

void FastKronHandle::getDistributedSizes(uint M, uint K, uint& gpuM, uint& gpuK) {
  //TODO: Should move to individual backends
  gpuM = M/cudaKernels.gpusInM_;
  gpuK = K/cudaKernels.gpusInK_;
}