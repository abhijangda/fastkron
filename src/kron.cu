#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <type_traits>
#include <thread>

#include <unordered_map>
#include <vector>
#include <iostream>
#include <sstream>
#include <limits>
#include <iomanip>
#include <cstring>

#include "utils.h"
#include "kron.h"
#include "thread_pool.h"
#include "device/otherkernels.cuh"


#define CUDA_LAST_ERROR do {                        \
  cudaError_t e = cudaGetLastError();               \
  if (e != cudaSuccess) {                           \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while (0)                                         \

#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define DIVUP(x,y) (((x) + (y) - 1)/((y)))
#define ROUNDUP(x,y) (DIVUP(x,y)*(y))
#define CUDA_WARP_SIZE 32

#define C_IN_REG
#define EVAL

//utils.h
static constexpr int log2(uint n) {return 31 - __builtin_clz(n);}
static constexpr int log2(int n) {return 31 - __builtin_clz(n);}

/*TODOs:
 1. Using fusion or not should be an environemnt flag
 2. Debug message environment flag*/
namespace env {
  static char DIST_COMM[] = "DIST_COMM";

  DistComm getDistComm() {
    char* val = getenv(DIST_COMM);
    if (val == nullptr) return DistComm::DistCommNone;
    if (strcmp(val, "P2P") == 0) return DistComm::P2P;
    if (strcmp(val, "NCCL") == 0) return DistComm::NCCL;
    std::cout << "Invalid value for DIST_COMM=" << val << std::endl;
    return DistComm::DistCommNone;
  }
}

template<>
struct std::hash<KronMatmulShape> {
  std::size_t operator()(const KronMatmulShape& k) const {
    return hash<uint>()(k.KronCols) ^ hash<uint>()(k.KronRows) ^ hash<uint>()(k.ColsA);
  }
};

//Map from Factor size and Number of factors to KernelInfos
static std::unordered_map<KronMatmulShape, std::vector<KernelInfo>> compiledKernels;

#include "kernel_defs.cuh"

// static_assert(sizeof(KronGemmKernels)/sizeof(KernelInfo) == NUM_TYPE_KERNELS * RowParallelismTy::Num * NUM_ROWS_MOD_TILE_IS_ZERO * NUM_KP_N_K_KERNELS * NUM_MAX_K_KERNELS*NUM_K_EQUALS_VAR*NUM_KPK_EQUALS_VAR);

template<typename T>
static int typeKernelIndex(T x) {
  if (std::is_same<T, float>::value)
    return 0;
  if (std::is_same<T, int>::value)
    return 1;
  if (std::is_same<T, double>::value)
    return 2;
}

/**Library entry points to launch cuda kernels**/

//Check N and K is a multiplication of KronMatCols and KronMatRows
static bool checkKronMatrixSizes(const uint NumKronMats, 
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

static bool checkDistributedKronSizes(const uint NumKronMats, 
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

KronMatmulShape maxCompiledColsA(KronMatmulShape shape) {
  while (compiledKernels.find(shape) == compiledKernels.end()) {
    shape.ColsA /= 2;
    if (shape.ColsA == 1) {
     break;
    }
  }
  // if (NumFusedKerns != -1 && shape.ColsA > 1) {
  //   while (compiledKernels.find(shape) != compiledKernels.end()) {
  //     bool found = false;
  //     for (auto info : compiledKernels.find(shape)->second) {
  //       if (info.NumFusedKerns == NumFusedKerns) {
  //         found = true;
  //         break;
  //       }
  //     }
  //     if (found) break;
  //     shape.ColsA /= 2;
  //   }
  // }

  // if (shape.ColsA == 1)
  //   std::cout << "Error: Cannot find compiled kernel\n";
  return shape;
}

uint maxFusedKernels(KronMatmulShape shape) {
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

KernelInfo selectKernel(KronMatmulShape shape) {
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

bool isValidKernel(KernelInfo& kernelInfo) {
  const uint NumThreads = kernelInfo.NumThreads;
  const uint KronRows = kernelInfo.KronRows;
  const uint KronCols = kernelInfo.KronCols;
  const uint CRegRows = kernelInfo.CRegRows;
  const uint CRegCols = kernelInfo.CRegCols;
  const uint MaxColsA = kernelInfo.MaxColsA;
  const uint TileKronCols = kernelInfo.TileKronCols;

  const uint ValidThreads = ((MaxColsA/KronRows)/CRegRows) * (TileKronCols/CRegCols);
  if (NumThreads != ROUNDUP(ValidThreads, CUDA_WARP_SIZE)) {
    std::cout << "Invalid kernel config " << kernelInfo << std::endl; 
    return false;
  }

  return true;
}

//Launch cuda kernels
template<typename T, uint NumFusedKerns>
cudaError_t generalSlicedMatmul(KernelInfo& kernelInfo, const uint kronIndex, 
                                T* x, T* kronMat[NumFusedKerns], T* kronGemmResult,
                                const uint M, const uint N, const uint K, 
                                const uint KronMatCols[NumFusedKerns], const uint KronMatRows[NumFusedKerns],
                                cudaStream_t stream) {
  cudaError_t status;
  
  if (!isValidKernel(kernelInfo)) abort();

  //Create the grid and thread block
  dim3 grid;
  dim3 block;
  grid = {
          (K/kernelInfo.MaxColsA) * DIVUP(KronMatCols[0], kernelInfo.TileKronCols),
          DIVUP(M, kernelInfo.TileRowsA),
          1
         };
  block = {
            kernelInfo.NumThreads, 
            1, 
            1
          };

  KernelParams<T, NumFusedKerns> params (M, N, K,
                                         KronMatRows, 
                                         KronMatCols, x, 
                                         kronMat, 
                                         kronGemmResult, 
                                         kronIndex);
  FusedParams<T, NumFusedKerns> fusedParams (M, N, K, kernelInfo.MaxColsA, KronMatRows, KronMatCols);
  // std::cout << "Invoking " << kernelInfo << std::endl;
  //Call kernel
  typedef void (*KronMatmulKernelTy)(KernelParams<T, NumFusedKerns>, FusedParams<T, NumFusedKerns>, DistributedParams<T>, dim3, dim3, cudaStream_t);
  KronMatmulKernelTy(kernelInfo.kernel)(params, fusedParams, DistributedParams<T>(), grid, block, stream);
  status = cudaGetLastError();
  CUDA_CHECK(status);
  return status;
}

//Launch cuda kernels
template<typename T, uint NumFusedKerns>
cudaError_t generalDistributedSlicedMatmul(KernelInfo& kernelInfo, const uint kronIndex, 
                                           T* x, T* kronMat[NumFusedKerns], T* kronGemmResult,
                                           const uint M, const uint N, const uint K, 
                                           const uint KronMatCols[NumFusedKerns], const uint KronMatRows[NumFusedKerns],
                                           DistributedParams<T> distParams, cudaStream_t stream) {
  cudaError_t status;
  
  if (!isValidKernel(kernelInfo)) abort();

  //Create the grid and thread block
  dim3 grid;
  dim3 block;
  
  grid = {
          (K/kernelInfo.MaxColsA) * DIVUP(KronMatCols[0], kernelInfo.TileKronCols),
          DIVUP(M, kernelInfo.TileRowsA),
          1
         };
  block = {
            kernelInfo.NumThreads, 
            1, 
            1
          };

  KernelParams<T, NumFusedKerns> params(M, N, K,
                                        KronMatRows, 
                                        KronMatCols, x, 
                                        kronMat, 
                                        kronGemmResult, 
                                        kronIndex);
  FusedParams<T, NumFusedKerns> fusedParams(M, N, K, kernelInfo.MaxColsA, KronMatRows, KronMatCols);

  //Call kernel
  //TODO: No need to have Type template (T) as part of Kernelparams and DistributedParams
  typedef void (*KronMatmulKernelTy)(KernelParams<T, NumFusedKerns>, FusedParams<T, NumFusedKerns>, DistributedParams<T>, dim3, dim3, cudaStream_t);
  KronMatmulKernelTy(kernelInfo.kernel)(params, fusedParams, distParams, grid, block, stream);
  status = cudaGetLastError();
  CUDA_CHECK(status);
  return status;
}

//TODO: These methods that take handle should be private methods of FastKronHandle
TunedKernelsSeries selectKernelSeries(FastKronHandle& handle, const uint NumKronMats,
                                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                                      bool distributedKernel) {
  uint MaxFusedKerns = handle.getUseFusion() ? maxFusedKernels(KronMatmulShape{KronMatCols[0], KronMatRows[0], K, M, 0}) : 1;
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
  
    bool DistributeToGPUs = distributedKernel && handle.distComm_ == DistComm::P2P && handle.gpusInK_ > 1 && (i == NumKronMats - 1);
    auto selectedKernel = selectKernel(KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                       prevTempN, M, NumFusedKerns, DistributeToGPUs});
    tunedSeries.push_back({selectedKernel, kronMat - NumFusedKerns, kronMat, prevTempN, 0.0f});
    prevTempN = currTempN;
  }

  return tunedSeries;
}

template<typename T, typename VecT>
cudaError_t singleGPUKronMatmul(FastKronHandle& handle, const uint NumKronMats, T* x, T* kronMats[], 
                                T* result,
                                uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], 
                                T* temp1, T* temp2, cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  if (result == NULL) return cudaErrorInvalidValue;
  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return cudaErrorInvalidValue;
  T* kronGemmResults[2] = {temp1, temp2};
  T* prevKronResult = x;
  T* currKronResult = kronGemmResults[0];
  //TODO: Assumes all factors are of same size and square shape
  TunedKernelsSeries kernelSeries;
  if (handle.tunedKernelSeries.size() > 0) {
    kernelSeries = handle.tunedKernelSeries;
  } else {
    kernelSeries = selectKernelSeries(handle, NumKronMats, M, N, K, 
                                      KronMatCols, KronMatRows, false);
  }
  //Use double buffering for writing result and using output 
  //of previous iteration as input to current
  uint prevTempN = K;
  uint currTempN;
  for (auto kernel : kernelSeries) {
    const uint kronMat = kernel.end;
    const uint NumFusedKerns = kernel.kernel.NumFusedKerns;
    T* krons[NumFusedKerns];
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
    switch(NumFusedKerns) {
      case 1:
        status = generalSlicedMatmul<T, 1>(selectedKernel, kronMat, prevKronResult,
                                           krons, currKronResult, M, currTempN, prevTempN,
                                           FusedKronMatCols, FusedKronMatRows,
                                           stream);
        break;
      case 2:
        status = generalSlicedMatmul<T, 2>(selectedKernel, kronMat, prevKronResult,
                                           krons, currKronResult, M, currTempN, prevTempN,
                                           FusedKronMatCols, FusedKronMatRows,
                                           stream);
        break;
      case 3:
        status = generalSlicedMatmul<T, 3>(selectedKernel, kronMat, prevKronResult,
                                           krons, currKronResult, M, currTempN, prevTempN,
                                           FusedKronMatCols, FusedKronMatRows,
                                           stream);
        break;
      case 4:
        status = generalSlicedMatmul<T, 4>(selectedKernel, kronMat, prevKronResult,
                                           krons, currKronResult, M, currTempN, prevTempN,
                                           FusedKronMatCols, FusedKronMatRows,
                                           stream);
        break;
      case 5:
        status = generalSlicedMatmul<T, 5>(selectedKernel, kronMat, prevKronResult,
                                           krons, currKronResult, M, currTempN, prevTempN,
                                           FusedKronMatCols, FusedKronMatRows,
                                           stream);
        break;
      default:
          std::cout << "Invalid number of fused kernels" << std::endl;
        status = cudaErrorInvalidValue;
    }
    
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

float minExecTimeOfSeries(uint M, uint K, const uint NumKronMats, 
                          uint KronMatCols[], uint KronMatRows[],
                          uint startKron, bool isDistributed,
                          TunedKernelsSeries& tunedKernels,
                          std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>> bestKernels) {
  if (startKron >= NumKronMats) return 0;
  bool distP2PStore = isDistributed;
  float minTime = std::numeric_limits<float>::max();
  TunedKernelsSeries minEpilogueKernels;
  TunedKernelFromStart minPrologueKernel;
  for (uint endKron = startKron; endKron < NumKronMats; endKron++) {
    const uint kronMat = endKron;
    //Include KronMats [startKron, ..., endKron]
    const uint NumFusedKerns = endKron - startKron + 1;
    uint FusedKronMatCols[NumFusedKerns];
    uint FusedKronMatRows[NumFusedKerns];
    for (int k = 0; k < NumFusedKerns; k++) {
      FusedKronMatCols[k] = KronMatCols[kronMat - k];
      FusedKronMatRows[k] = KronMatRows[kronMat - k];
    }

    //TODO: Change tempN to tempK everywhere else
    uint tempK = K;
    for (int reverseKron = NumKronMats - 1; reverseKron > endKron; reverseKron--) {
      tempK = (tempK/KronMatRows[reverseKron])*KronMatCols[reverseKron];
    }

    KronMatmulShape shape = KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                            tempK, M, NumFusedKerns, 
                                            distP2PStore && startKron == 0};
    if (bestKernels.find(shape) == bestKernels.end()) continue;
    auto iter = bestKernels.find(shape);
    TunedKernelsSeries epilogueKernels;
    float kernelTime = iter->second.second;
    float epilogueTime = minExecTimeOfSeries(M, K, NumKronMats, KronMatCols,
                                             KronMatRows, endKron + 1, isDistributed, 
                                             epilogueKernels, bestKernels);
    if (minTime > kernelTime + epilogueTime) {
      minTime = kernelTime + epilogueTime;
      minEpilogueKernels = epilogueKernels;
      minPrologueKernel = TunedKernelFromStart(iter->second.first, 
                                               startKron, endKron, tempK, kernelTime);
    }
  }
  tunedKernels = minEpilogueKernels;
  tunedKernels.push_back(minPrologueKernel);

  assert(minTime < std::numeric_limits<float>::max());
  return minTime;
}

//TODO: Create another autotuning object?
template<typename T>
cudaError_t singleGPUAutotune(FastKronHandle& handle, const uint NumKronMats, T* x, T* kronMats[],
                              uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                              T* temp1, T* temp2,
                              bool isDistributed, DistributedParams<T> distParams,
                              std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>>& bestKernels,
                              cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  T* kronGemmResults[2] = {(T*)temp1, (T*)temp2};
  //For performance eval we do not need these to contain any value
  T* prevKronResult = kronGemmResults[0];
  T* currKronResult = kronGemmResults[1];
  //TODO: Assumes all factors are of same size and square shape
  // const uint MaxFusedKerns = handle.getUseFusion() ? 
  //                            maxFusedKernels(KronMatmulShape{KronMatCols[0], KronMatRows[0], K}) : 1;
  //Use double buffering for writing result and using output 
  //of previous iteration as input to current
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  
  //A KronMat is a series of SlicedMats
  //We need to get best kernel for all contiguous SlicedMats
  for (uint startKron = 0; startKron < NumKronMats; startKron++) {
  for (uint endKron = startKron; endKron < NumKronMats; endKron++)   {
    const uint kronMat = endKron;
    //KronMats[startKron, ..., endKron] including endKron
    const uint NumFusedKerns = endKron - startKron + 1;
    T* krons[NumFusedKerns];
    uint FusedKronMatCols[NumFusedKerns];
    uint FusedKronMatRows[NumFusedKerns];
    for (int k = 0; k < NumFusedKerns; k++) {
      krons[k] = kronMats[kronMat - k];
      FusedKronMatCols[k] = KronMatCols[kronMat - k];
      FusedKronMatRows[k] = KronMatRows[kronMat - k];
    }
    uint tempN = K;
    for (int reverseKron = NumKronMats - 1; reverseKron > endKron; reverseKron--) {
      tempN = (tempN/KronMatRows[reverseKron])*KronMatCols[reverseKron];
    }
    uint outTempN = (tempN/KronMatRows[endKron])*KronMatCols[endKron];
    // std::cout << "endKron " << endKron << " startKron " << startKron << " tempN " << tempN << std::endl;
    bool distP2PStore = isDistributed && startKron == 0;
    cudaError_t status;
    KronMatmulShape shape = KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                            tempN, M, NumFusedKerns, distP2PStore};
    if (bestKernels.find(shape) != bestKernels.end()) {
      continue;
    }
    if (!handle.getUseFusion() and NumFusedKerns > 1) continue;
    KernelInfo bestKernel;
    float minTime = std::numeric_limits<float>::max();
    const uint runs = 5;
    const uint warmups = 2;
    std::cout << "Tuning for shape "  << shape << std::endl;
    for (auto shapeAndKernels : compiledKernels) {
      if (!shapeAndKernels.first.sameKronSize(shape)) continue;
      for (auto kernel : shapeAndKernels.second) {
        if (!kernel.canCompute(shape)) continue;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int r = 0; r < warmups + runs; r++) {
          if (r == warmups) CUDA_CHECK(cudaEventRecord(start, stream));
          if (distP2PStore) {
            switch (NumFusedKerns) {
              case 1:
                status = generalDistributedSlicedMatmul<T, 1>(kernel, endKron, prevKronResult, 
                                                              krons, currKronResult, M, outTempN, tempN, 
                                                              FusedKronMatCols, FusedKronMatRows, 
                                                              distParams, stream);
                break;
              case 2:
                status = generalDistributedSlicedMatmul<T, 2>(kernel, endKron, prevKronResult, 
                                                              krons, currKronResult, M, outTempN, tempN,
                                                              FusedKronMatCols, FusedKronMatRows, 
                                                              distParams, stream);
                break;
              case 3:
                status = generalDistributedSlicedMatmul<T, 3>(kernel, endKron, prevKronResult, 
                                                              krons, currKronResult, M, outTempN, tempN,
                                                              FusedKronMatCols, FusedKronMatRows, 
                                                              distParams, stream);
                break;
              case 4:
                status = generalDistributedSlicedMatmul<T, 4>(kernel, endKron, prevKronResult, 
                                                              krons, currKronResult, M, outTempN, tempN,
                                                              FusedKronMatCols, FusedKronMatRows, 
                                                              distParams, stream);
                break;
              case 5:
                status = generalDistributedSlicedMatmul<T, 5>(kernel, endKron, prevKronResult, 
                                                              krons, currKronResult, M, outTempN, tempN, 
                                                              FusedKronMatCols, FusedKronMatRows, 
                                                              distParams, stream);
                break;
            }
          } else {
            switch(NumFusedKerns) {
              case 1:
                status = generalSlicedMatmul<T, 1>(kernel, endKron, prevKronResult,
                                                  krons, currKronResult, M, outTempN, tempN, 
                                                  FusedKronMatCols, FusedKronMatRows,
                                                  stream);
                break;
              case 2:
                status = generalSlicedMatmul<T, 2>(kernel, endKron, prevKronResult,
                                                  krons, currKronResult, M, outTempN, tempN, 
                                                  FusedKronMatCols, FusedKronMatRows,
                                                  stream);
                break;
              case 3:
                status = generalSlicedMatmul<T, 3>(kernel, endKron, prevKronResult,
                                                  krons, currKronResult, M, outTempN, tempN,
                                                  FusedKronMatCols, FusedKronMatRows,
                                                  stream);
                break;
              case 4:
                status = generalSlicedMatmul<T, 4>(kernel, endKron, prevKronResult,
                                                  krons, currKronResult, M, outTempN, tempN,
                                                  FusedKronMatCols, FusedKronMatRows,
                                                  stream);
                break;
              case 5:
                status = generalSlicedMatmul<T, 5>(kernel, endKron, prevKronResult,
                                                  krons, currKronResult, M, outTempN, tempN,
                                                  FusedKronMatCols, FusedKronMatRows,
                                                  stream);
                break;
              default:
                  std::cout << "Invalid number of fused kernels" << std::endl;
                status = cudaErrorInvalidValue;
            }
          }
          // if (status != cudaSuccess) break;
        }
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        
        if (status != cudaSuccess)
          std::cout << "Error: " << cudaGetErrorString(status) << " for " << kernel << " tempN " << tempN << std::endl;
        float kernelTime;
        CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, end));
        std::cout << std::fixed << std::setprecision(2) << 
                     kernel << " runs in " << (kernelTime/runs) << " ms " << std::endl;
        if (kernelTime < minTime) {
          bestKernel = kernel;
          minTime = kernelTime;
        }
        if (status != cudaSuccess) return status;
      }
    }

    if (minTime < std::numeric_limits<float>::max()) {
      std::cout << std::fixed << std::setprecision(2) <<
                   "Best kernel for " << shape << ": " << bestKernel << " runs in " << (minTime/runs) << " ms" << std::endl;
      bestKernels.emplace(std::make_pair(shape, std::make_pair(bestKernel, minTime/runs)));
    }
  }}

  return cudaSuccess;
}

template<typename T>
cudaError_t autotune(FastKronHandle& handle, const uint NumKronMats, T* x, T* kronMats[], 
                     uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                     cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return cudaErrorInvalidValue;

  std::cout << "N " << N << " K " << K << " KronMatCols[0] " << KronMatCols[0] << " KronMatRows[0] " << KronMatRows[0] << std::endl;
  float minTime = 0;
  if (!handle.isDistributed_) {
    //TODO: temp1_ and temp2_ declaration/allocation is same for both cases
    T* temp1_, *temp2_;
    size_t resultSize = 0, tempSize = 0;
    kronGeMMSizes(handle, NumKronMats, M, N, K, KronMatCols, KronMatRows, 
                  &resultSize, &tempSize);  
    std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>> bestKernels;
    CUDA_CHECK(cudaMalloc(&temp1_, tempSize * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&temp2_, tempSize * sizeof(T)));
    singleGPUAutotune(handle, NumKronMats, x, kronMats, M, N, K, KronMatCols, KronMatRows, 
                      (T*)temp1_, (T*)temp2_, false, DistributedParams<T>(), 
                      bestKernels, stream);
    std::cout << "Finding min execution time of the series" << std::endl;
    TunedKernelsSeries tunedKernels;
    minTime = minExecTimeOfSeries(M, K, NumKronMats,
                                  KronMatCols, KronMatRows, 0, false,
                                  tunedKernels, bestKernels);
    handle.tunedKernelSeries = tunedKernels;
    CUDA_CHECK(cudaFree(temp1_));
    CUDA_CHECK(cudaFree(temp2_));
  } else {
    if (!checkDistributedKronSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows, handle.perGPUKronBatch_, handle.gpusInK_))
      return cudaErrorInvalidValue;

    //In distributed case run every LocalKron series on a single GPU
    CUDA_CHECK(cudaSetDevice(0));
    T* temp1_[handle.numGPUs_], *temp2_[handle.numGPUs_];
    size_t resultSize = 0, tempSize = 0;
    kronGeMMSizes(handle, NumKronMats, M, N, K, KronMatCols, KronMatRows, 
                  &resultSize, &tempSize);
    for (int g = 0; g < handle.numGPUs_; g++) {
      CUDA_CHECK(cudaMalloc(&temp1_[g], tempSize * sizeof(T)));
      CUDA_CHECK(cudaMalloc(&temp2_[g], tempSize * sizeof(T)));
    }
    minTime = std::numeric_limits<float>::max();
    uint gpuM, gpuK;
    handle.getDistributedSizes(M, K, gpuM, gpuK);
    uint prevTempN = gpuK;
    //TODO: This loop is really common and should be a macro?
    std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>> bestKernels;

    uint bestMaxLocalKrons = 1;
    TunedKernelsSeries minKernelSeries;
    for (uint MaxLocalKrons = 1; MaxLocalKrons < NumKronMats; MaxLocalKrons += 1) {
    uint seriesTime = 0;
    TunedKernelsSeries tunedKernelSeries;
    
    for (uint i = 0; i < NumKronMats; i += MaxLocalKrons) {
      const uint kronMat = NumKronMats - i - 1;
      const uint LocalKrons = min(MaxLocalKrons, NumKronMats - i);
      uint currTempN = prevTempN;
      uint LocalKronMatCols[LocalKrons];
      uint LocalKronMatRows[LocalKrons];
      for (int k = 0; k < LocalKrons; k++) {
        LocalKronMatCols[k] = KronMatCols[kronMat - k];
        LocalKronMatRows[k] = KronMatRows[kronMat - k];
        currTempN = (currTempN/LocalKronMatRows[k])*LocalKronMatCols[k];
      }
      
      T** gpuResults = (T**)temp2_;
      int prevFullK = prevTempN * handle.gpusInK_;
      int currFullN = currTempN * handle.gpusInK_;
      DistributedParams<T> distParams(0, 0, handle.gpusInK_, prevFullK, currFullN, 
                                      prevFullK, currFullN, LocalKronMatCols, LocalKronMatRows, LocalKrons);
      distParams.updateGPUResults(gpuResults);
      singleGPUAutotune(handle, LocalKrons, x, kronMats, gpuM, currTempN, prevTempN, 
                        LocalKronMatCols, LocalKronMatRows, temp1_[0], temp2_[0],
                        handle.isDistributed_ && handle.distComm_ == DistComm::P2P, 
                        distParams, bestKernels, stream);
      TunedKernelsSeries tunedKernels;
      seriesTime += minExecTimeOfSeries(gpuM, prevTempN, LocalKrons,
                                     LocalKronMatCols, LocalKronMatRows, 0,
                                     handle.isDistributed_ && handle.distComm_ == DistComm::P2P,
                                     tunedKernels, bestKernels);

      for (auto tunedKernel : tunedKernels) {
        tunedKernel.start += kronMat + 1 - LocalKrons;
        tunedKernel.end   += kronMat + 1 - LocalKrons;
        tunedKernelSeries.insert(tunedKernelSeries.begin(), tunedKernel);
      }
    }
    
    if (seriesTime < minTime) {
      minTime = seriesTime;
      handle.tunedKernelSeries = tunedKernelSeries;
      handle.perGPUKronBatch_ = MaxLocalKrons;
    }
    }

    for (int g = 0; g < handle.numGPUs_; g++) {
      CUDA_CHECK(cudaFree(temp1_[g]));
      CUDA_CHECK(cudaFree(temp2_[g]));
    }
  }

  std::cout <<"Minimum Time " << minTime << " through kernels: " << std::endl;
  for (auto iter = handle.tunedKernelSeries.rbegin(); iter != handle.tunedKernelSeries.rend(); iter++) {
    std::cout << "  " << (*iter) << std::endl;
  }
  return cudaSuccess;
}

extern cudaError_t kronSGEMMTune(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], 
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                                 cudaStream_t stream) {
  return autotune<float>(handle, NumKronMats, x, kronMats,
                         M, N, K, KronMatCols, KronMatRows,
                         stream);
}

cudaError_t kronDGEMMTune(FastKronHandle& handle, const uint NumKronMats, double* x, double* kronMats[], 
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream) {
  return autotune<double>(handle, NumKronMats, x, kronMats, 
                          M, N, K, KronMatCols, KronMatRows,
                          stream);
}

cudaError_t kronIGEMMTune(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[],
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream) {
  return autotune<int>(handle, NumKronMats, x, kronMats,
                       M, N, K, KronMatCols, KronMatRows,
                       stream);
}

void thread_barrier_wait(pthread_barrier_t* barrier) {
  int s = pthread_barrier_wait(barrier);
  assert (s == 0 || s == PTHREAD_BARRIER_SERIAL_THREAD);
}

template<typename T, typename VecT>
void perGPUKronMatmul(ThreadArgs* thArgs) {
  // ThreadArgs<T>& thArgs = *(ThreadArgs<T>*)arg;

  FastKronHandle& handle = *thArgs->handle;
  uint NumKronMats = thArgs->NumKronMats;
  T* x = (T*)thArgs->x;
  T** kronMats = (T**)thArgs->kronMats;
  T** results = (T**)thArgs->result;
  T** temp1 = (T**)thArgs->temp1;
  T** temp2 = (T**)thArgs->temp2;
  uint M = thArgs->M;
  uint N = thArgs->N;
  uint K = thArgs->K;
  uint *KronMatCols = thArgs->KronMatCols;
  uint *KronMatRows = thArgs->KronMatRows;
  cudaStream_t* stream = thArgs->stream;
  uint gr = thArgs->gpuRow;
  uint gc = thArgs->gpuCol;
  uint gpusInM_ = thArgs->gpusInM_;
  uint gpusInK_ = thArgs->gpusInK_; 
  uint currTempN;
  uint g = gr * gpusInK_ + gc;
  CUDA_CHECK(cudaSetDevice(g));

  cudaError_t status;
  
  //Temporaries are swaped after every slicedMatmul
  //TODO: User supplied result should be used as a temp and the final results are written in it
  //TODO: What if Rows are not multiple of GPUs in Rows
  T* innerResults[2] = {(T*)temp1[g], (T*)temp2[g]};
  // std::cout << "handle.gpuM_ " << handle.gpuM_ << " handle.gpuK_ " <<handle.gpuK_ << " gpusInCols " << gpusInCols << " gpusInRows " << gpusInRows << " K " << K << std::endl;
  T* innerPrevResult;
  T* innerCurrResult;
  uint gpuM, gpuK;
  handle.getDistributedSizes(M, K, gpuM, gpuK);
  uint prevTempN = gpuK;

  uint startGpuM = gpuM * gr;
  // const uint gpuM = min(gpuM, M - startGpuM);
  //For first slicedMatmul, x is the input
  innerPrevResult = x;
  innerCurrResult = innerResults[0];
  // if (g == 1) {printf("627\n"); printGPUArray<float>(handle.gpuM_, handle.gpuK_, innerPrevResult, stream[g]);}
  CUDA_CHECK(cudaSetDevice(g));

  for (uint io = 0; io < NumKronMats; io += handle.perGPUKronBatch_) {
    // if (io == 0) {
    // } else {
    //   innerPrevResult = innerResults[0];
    //   innerCurrResult = innerResults[1];
    // }

    // TODO:
    // if (uvaColsX == K) {
    //   innerResults[0] = &kronGemmResults[0][startOutofCoreRows * K];
    //   innerResults[1] = &kronGemmResults[1][startOutofCoreRows * K];
    //   innerPrevResult = &x[startOutofCoreRows * K];
    //   innerCurrResult = innerResults[0];
    // }
    uint KronMulBatchSize = min(handle.perGPUKronBatch_, NumKronMats - io);
    uint MaxI = io + KronMulBatchSize;
    {//uint gpuColPart = gc * handle.gpuK_; gpuColPart < K; gpuColPart += handle.gpuK_ * gpusInK_) {
      //Copy outerPrevResult to innerPrevResult
      // if (uvaColsX < K) {
      //   // CUDA_CHECK(cudaDeviceSynchronize());
      //   // printf("copyXtoUVAX\n");
      //   dim3 grid = {outOfCoreRows, 1,1};
      //   dim3 block = {256, 1, 1};
      //   copyXtoUVAX<T, VecT, 256><<<grid, block, 0, stream[g]>>>(M, N, K, KronMatRows[io], KronMatCols[io], innerPrevResult, outOfCoreRows, uvaColsX,
      //                                                             &outerPrevResult[startOutofCoreRows * K], uvaPart/uvaColsX);
      //   // CUDA_CHECK(cudaDeviceSynchronize());
      //   // printf("Done\n");
      // }
      const uint endKron = NumKronMats - io - KronMulBatchSize;
      
      currTempN = prevTempN;

      TunedKernelsSeries kernelSeries;
      uint LocalKronCols[KronMulBatchSize];
      uint LocalKronRows[KronMulBatchSize];
      for (int i = KronMulBatchSize - 1; i >= 0 ; i--) {
        LocalKronCols[i] = KronMatCols[NumKronMats - MaxI + i];
        LocalKronRows[i] = KronMatRows[NumKronMats - MaxI + i];
        currTempN = (currTempN/LocalKronRows[i])*LocalKronCols[i];
      }

      if (handle.tunedKernelSeries.size() > 0) {
        for (auto tunedKernel : handle.tunedKernelSeries) {
          if (tunedKernel.start >= endKron  and tunedKernel.end < endKron + KronMulBatchSize) {
            kernelSeries.insert(kernelSeries.begin(), tunedKernel);
          }
        }
      } else {
        auto localSeries = selectKernelSeries(handle, KronMulBatchSize, gpuM, gpuK, gpuK, 
                                              LocalKronCols, LocalKronRows, true);
        for (auto& kernel : localSeries) {
          kernel.end += endKron;
        }
        kernelSeries = localSeries;
      }
      int prevFullK = prevTempN * handle.gpusInK_;
      int currFullN = currTempN * handle.gpusInK_;
      DistributedParams<T> distParams(gr, gc, handle.gpusInK_, 
                                      prevFullK, currFullN,
                                      prevTempN, currTempN, LocalKronCols, LocalKronRows, KronMulBatchSize);
      uint slicedMuls = 0;
      bool ncclRecvInResult = false;
      for (auto kernel : kernelSeries) {
        //TODO: probably will need to change for fused kernels
        const uint NumFusedKerns = kernel.kernel.NumFusedKerns;
        
        T* krons[NumFusedKerns];
        uint kronCols[NumFusedKerns];
        uint kronRows[NumFusedKerns];
        
        currTempN = prevTempN;
        for (int kk = 0; kk < NumFusedKerns; kk++) {
          krons[kk] = kronMats[g * NumKronMats + kernel.end - kk];
          kronRows[kk] = KronMatRows[kernel.end - kk];
          kronCols[kk] = KronMatCols[kernel.end - kk];
          currTempN = (currTempN/kronRows[kk])*kronCols[kk];
        }

        if (slicedMuls == KronMulBatchSize - 1) {
          CUDA_CHECK(cudaStreamSynchronize(stream[g]));
          thread_barrier_wait(thArgs->barrier);
        }
        
        if (kernel.end - NumFusedKerns + 1 == 0) {
          if (handle.distComm_ == DistComm::P2P or handle.gpusInK_ == 1)
            innerCurrResult = results[g];
          else
            ncclRecvInResult = true;
        } 
        
        T** gpuTempResults;
        if (innerCurrResult == innerResults[0]) {
          gpuTempResults = (T**)temp1;
        } else if (innerCurrResult == innerResults[1]) {
          gpuTempResults = (T**)temp2;
        } else if (innerCurrResult == results[g]) {
          gpuTempResults = (T**)results;
        }
        
        T* gpuResults[handle.gpusInK_];
        for (int _gc = 0; _gc < handle.gpusInK_; _gc++) {
          gpuResults[_gc] = gpuTempResults[gr * handle.gpusInK_ + _gc];
        }
        distParams.updateGPUResults(gpuResults);

        //TODO: a single switch case for FusedKernels?
        cudaError_t status;
        switch (NumFusedKerns) {
          case 1:
            status = generalDistributedSlicedMatmul<T, 1>(kernel.kernel, kernel.end, innerPrevResult, 
                                                          krons, innerCurrResult, gpuM, currTempN, 
                                                          prevTempN, kronCols, kronRows, distParams, 
                                                          stream[g]);
            break;
          case 2:
          //TODO: for all other cases
            status = generalDistributedSlicedMatmul<T, 2>(kernel.kernel, kernel.end, innerPrevResult, 
                                                          krons, innerCurrResult, gpuM, gpuK, gpuK, 
                                                          kronCols, kronRows, distParams, stream[g]);
            break;
          case 3:
            status = generalDistributedSlicedMatmul<T, 3>(kernel.kernel, kernel.end, innerPrevResult, 
                                                          krons, innerCurrResult, gpuM, gpuK, gpuK, 
                                                          kronCols, kronRows, distParams, stream[g]);
            break;
          case 4:
            status = generalDistributedSlicedMatmul<T, 4>(kernel.kernel, kernel.end, innerPrevResult, 
                                                          krons, innerCurrResult, gpuM, gpuK, gpuK, 
                                                          kronCols, kronRows, distParams, stream[g]);
            break;
          case 5:
            status = generalDistributedSlicedMatmul<T, 5>(kernel.kernel, kernel.end, innerPrevResult, 
                                                          krons, innerCurrResult, gpuM, gpuK, gpuK, 
                                                          kronCols, kronRows, distParams, stream[g]);
            break;
        }
        assert(status == cudaSuccess);        
        CUDA_CHECK(cudaStreamSynchronize(stream[g]));
        
        // if (gc == 0 and kernel.end == 1) {
        //   printGPUArray(handle.gpuM_, handle.gpuK_, 128.0f*128.0f, innerCurrResult, stream[g]);
        // }
        // if (gc == 0) printf("slicedMuls %d innerCurrResult %p innerPrevResult %p\n", slicedMuls, innerCurrResult, innerPrevResult);
        // if (status != cudaSuccess) goto end;
        prevTempN = currTempN;
        //Double/ring/circular buffer previous result and new result
        innerPrevResult = innerCurrResult;
        if (innerPrevResult == innerResults[0]) {
          innerCurrResult = innerResults[1];
        } else if (innerPrevResult == innerResults[1]) {
          innerCurrResult = innerResults[0];
        }
        slicedMuls++;
      }

      CUDA_CHECK(cudaStreamSynchronize(stream[g]));
      
      thread_barrier_wait(thArgs->barrier);

      if (handle.distComm_ == DistComm::NCCL && handle.gpusInK_ > 1) {
        size_t resultSize = 0, tempSize = 0;
        if (ncclRecvInResult)
          innerCurrResult = results[g];
        kronGeMMSizes(handle, NumKronMats, M, N, K, KronMatCols, KronMatRows, 
                      &resultSize, &tempSize);
        T* sendTemp = temp1[g] + tempSize/2;
        T* recvTemp = temp2[g] + tempSize/2;
        //Call we want to use NCCL Send/Recv
        {
          const uint SliceRows = gpuM;
          const uint SliceCols = currTempN/handle.gpusInK_;
          const size_t sendRecvSize = SliceRows * SliceCols;
          const uint startRow = 0;
          const uint startCol = gc * SliceCols;
          matrixSlice(gpuM, currTempN, innerPrevResult, 
                      startRow, startCol, SliceRows, SliceCols,
                      recvTemp, stream[g], g, io, true);
          dim3 grid = {gpuM, 1,1};
          dim3 block = {256, 1, 1};
          storeGPUTile<T, VecT, 256><<<grid, block, 0, stream[g]>>>(M, currTempN*handle.gpusInK_, prevTempN*handle.gpusInK_,
                                                                    KronMatRows[0], KronMatCols[0], gc, handle.gpusInK_,
                                                                    recvTemp, gpuM, currTempN,
                                                                    innerCurrResult, gc, KronMulBatchSize, io, distParams, false);
          // if (g == 0) {
          //   std::cout << "io " << io << " SliceCols " << SliceCols << std::endl;
          //   float val;
          //   if (io == 0) val = 64.0f;
          //   else if (io == 1) val = 64.0f * 64.0f;
          //   else if (io == 2) val = 64.0f * 64.0f * 64.0f;
          //   else if (io == 3) val = 64.0f * 64.0f * 64.0f * 64.0f;
          //   if (io <= 0)
          //   printGPUArray<float>(handle.gpuM_, SliceCols, val,
          //     (float*)innerCurrResult, stream[g]);
          // }
          CUDA_CHECK(cudaStreamSynchronize(stream[g]));
        }

        //All GPUs with the same gr share their intermediates
        for (int dst = 0; dst < handle.gpusInK_; dst++) {
          const uint SliceRows = gpuM;
          const uint SliceCols = currTempN/handle.gpusInK_;
          const size_t sendRecvSize = SliceRows * SliceCols;
          if (dst == gc) {
            for (int src = 0; src < handle.gpusInK_; src++) {
              // printf("g %d dst %d src %d\n", g, dst, src);
              if (src == dst) {
              } else {
                NCCLCHECK(ncclRecv(recvTemp, sendRecvSize, ncclFloat, gr * handle.gpusInK_ + src, handle.ncclComms[g], stream[g]));
                CUDA_CHECK(cudaStreamSynchronize(stream[g]));
                dim3 grid = {gpuM, 1,1};
                dim3 block = {256, 1, 1};
                storeGPUTile<T, VecT, 256><<<grid, block, 0, stream[g]>>>(M, currTempN*handle.gpusInK_, prevTempN*handle.gpusInK_,
                                                                          KronMatRows[0], KronMatCols[0], gc, handle.gpusInK_,
                                                                          recvTemp, gpuM, currTempN,
                                                                          innerCurrResult, src, KronMulBatchSize, io, distParams, false);
                CUDA_CHECK(cudaStreamSynchronize(stream[g]));
              }
            }
          } else {
            const uint startRow = 0;
            const uint startCol = dst * SliceCols;
            matrixSlice(gpuM, currTempN, innerPrevResult, 
                        startRow, startCol, SliceRows, SliceCols,
                        sendTemp, stream[g], g, io);
            CUDA_CHECK(cudaStreamSynchronize(stream[g]));
            // if (g == 1 && dst == 0) {
            //    printGPUArray<float>(SliceRows, SliceCols, (float*)handle.sendTemps_[g], stream[g]);
            //    printf("699 dst %d g %d\n", dst, g);
            // }
            NCCLCHECK(ncclSend(sendTemp, sendRecvSize, ncclFloat, gr * handle.gpusInK_ + dst, handle.ncclComms[g], stream[g]));
            CUDA_CHECK(cudaStreamSynchronize(stream[g]));
          }
        }

        innerPrevResult = innerCurrResult;
        if (innerPrevResult == innerResults[0]) {        
          innerCurrResult = innerResults[1];
        } else if (innerPrevResult == innerResults[1]) {
          innerCurrResult = innerResults[0];
        }
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream[g]));
    thread_barrier_wait(thArgs->barrier);
  }

  end:
  thArgs->threadResult = {status, (void*)innerPrevResult};
}

template<typename T, typename VecT>
cudaError_t distributedKronMatmul(FastKronHandle& handle, const uint NumKronMats, T* x[], T* kronMats[], T* result[],
                                  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
                                  cudaStream_t streams[]) {
  uint gpuM, gpuK;
  handle.getDistributedSizes(M, K, gpuM, gpuK);
  if (!checkDistributedKronSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows, handle.perGPUKronBatch_, handle.gpusInK_))
    return cudaErrorInvalidValue;

  if (result == NULL)                        return cudaErrorInvalidValue;
  if (M % gpuM != 0)                         return cudaErrorInvalidValue;
  if (NumKronMats < handle.perGPUKronBatch_) return cudaErrorInvalidValue;

  const uint batchedKronMuls = handle.perGPUKronBatch_;

  thread_pool<ThreadArgs*>::task tasks[handle.numGPUs_];
  ThreadArgs threadArgs[handle.numGPUs_];

  for (uint thread = 0; thread < handle.numGPUs_; thread++) {
    ThreadArgs args = ThreadArgs(
      &handle,
      NumKronMats,
      (void*)x[thread],
      (void**)kronMats,
      (void**)result,
      M, N, K,
      &KronMatCols[0],
      &KronMatRows[0],
      (void**)temp1, (void**)temp2,
      streams,
      thread/handle.gpusInK_,
      thread % handle.gpusInK_,
      handle.gpusInM_,
      handle.gpusInK_,
      &handle.barriers_[thread/handle.gpusInK_]
    );

    threadArgs[thread] = args;
    tasks[thread] = thread_pool<ThreadArgs*>::task(perGPUKronMatmul<T, VecT>, &threadArgs[thread]);
  }

  handle.threads_->execute_tasks(tasks);
  handle.threads_->join_tasks();

  cudaError_t status;
  for (uint thread = 0; thread < handle.numGPUs_; thread++) {
    status = threadArgs[thread].threadResult.status;
    // result[thread] =(T*)threadArgs[thread].threadResult.result;
  }

  return status;
}

/**************************************************
          Library Functions
***************************************************/
cudaError_t kronSGEMM(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float* result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float* temp1, float* temp2,
                      cudaStream_t stream) {
  return singleGPUKronMatmul<float, float4>(handle, NumKronMats, x, kronMats, result,
                                            M, N, K, KronMatCols, KronMatRows, temp1, temp2, 
                                            stream);
}

cudaError_t kronIGEMM(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], int* result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], int* temp1, int* temp2,
                      cudaStream_t stream) {
  return singleGPUKronMatmul<int, int4>(handle, NumKronMats, x, kronMats, result, 
                                        M, N, K, KronMatCols, KronMatRows, temp1, temp2,
                                        stream);
}

cudaError_t kronDGEMM(FastKronHandle& handle, const uint NumKronMats, double* x, double* kronMats[], double* result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], double* temp1, double* temp2,
                      cudaStream_t stream) {
  return singleGPUKronMatmul<double, double4>(handle, NumKronMats, x, kronMats, result, 
                                              M, N, K, KronMatCols, KronMatRows, temp1, temp2,
                                              stream);
}


cudaError_t kronSGEMMOutofCore(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream) {
  // return singleGPUOutOfCoreKronMatmul<float, float4>(handle, NumKronMats, x, kronMats, result, 
  //                                                    M, N, K, KronMatCols, KronMatRows, stream);
}

// cudaError_t kronSGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
//   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]) {
//   return singleGPUOutOfCoreKronMatmul<float, float4>(handle, NumKronMats, x, kronMats, result, 
//                                                      M, N, K, KronMatCols, KronMatRows, stream);
// }

// cudaError_t kronIGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], int** result,
//   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]) {
//   return singleGPUOutOfCoreKronMatmul<int, int4>(handle, NumKronMats, x, kronMats, result, 
//                                                  M, N, K, KronMatCols, KronMatRows, stream);
// }

cudaError_t kronDistributedSGEMM(FastKronHandle& handle, const uint NumKronMats, float* x[], float* kronMats[], float* result[],
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
                                 cudaStream_t streams[]) {
  return distributedKronMatmul<float, float4>(handle, NumKronMats, x, kronMats, result, M, N, K, 
                                              KronMatCols, KronMatRows, temp1, temp2, streams);
}

cudaError_t kronGeMMSizes(FastKronHandle& handle, const uint NumKronMats, uint M, uint N, uint K, 
                          uint KronMatCols[], uint KronMatRows[], size_t* resultSize, size_t* tempSize) {
  if (resultSize == nullptr) return cudaErrorInvalidValue;
  if (tempSize   == nullptr) return cudaErrorInvalidValue;
  uint gpuM, gpuK;

  if (handle.isDistributed_) {
    if (!checkDistributedKronSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows, 
                                   handle.perGPUKronBatch_, handle.gpusInK_))
      return cudaErrorInvalidValue;
    gpuM = M/handle.gpusInM_;
    gpuK = K/handle.gpusInK_;
  } else {
    if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
      return cudaErrorInvalidValue;
    gpuM = M;
    gpuK = K;
  }

  size_t tempN = gpuK;
  size_t maxTempN = tempN;
  for (int i = NumKronMats - 1; i >= 0; i--) {
    tempN = (tempN/KronMatRows[i])*KronMatCols[i];
    if (maxTempN < tempN)
      maxTempN = tempN;
  }

  *tempSize   = gpuM * maxTempN;
  if (handle.isDistributed_ and handle.distComm_ == DistComm::NCCL)
    //Include size of send and recv buffers 
    *tempSize = (*tempSize) * 2;
  *resultSize = gpuM * tempN;

  return cudaSuccess;
}

void FastKronHandle::getDistributedSizes(uint M, uint K, uint& gpuM, uint& gpuK) {
  gpuM = M/gpusInM_;
  gpuK = K/gpusInK_;
}

uint getYColumns(uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  size_t tempN = K;
  size_t maxTempN = tempN;
  for (int i = 0; i < NumKronMats; i++) {
    tempN = (tempN/KronMatRows[i])*KronMatCols[i];
    if (maxTempN < tempN)
      maxTempN = tempN;
  }

  return tempN;
}

template<typename T> cudaError_t FastKronHandle_allocDistributedX(FastKronHandle& handle, T* dX[], T* hX, uint M, uint K) {
  //TODO: Make FastKronError type
  if (!handle.isDistributed_) return cudaErrorInvalidValue;
  uint gpuM, gpuK;
  handle.getDistributedSizes(M, K, gpuM, gpuK);
  //TODO: Check that hX is on host memory
  T* gpuHostX = new T[((size_t)gpuM) * ((size_t)gpuK)];
  std::cout << "Distributing X to all GPUs "<<std::endl;
  // std::cout << handle.gpuM_ << "  " << handle.gpuK_ << "  " << sizeof(T) << std::endl;
  for (int g = 0; g < handle.numGPUs_; g++) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMalloc(&dX[g], sizeof(T) * gpuM * gpuK));
  }

  for(int gr = 0; gr < handle.gpusInM_; gr++) {
    for (uint gc = 0; gc < handle.gpusInK_; gc++) {
      const uint g = gr * handle.gpusInK_ + gc;
      // std::cout << "g " << g << " gr " <<gr << " gc " << gc << std::endl;
      CUDA_CHECK(cudaSetDevice(g));
      uint startGpuM = gpuM * gr;
      uint startGpuK = gpuK * gc;
        
      for (uint m = 0; m < gpuM; m++) {
        std::memcpy(&gpuHostX[m * gpuK], &hX[(startGpuM+m)*K + startGpuK], sizeof(T)*gpuK);
      }
      CUDA_CHECK(cudaMemcpy(dX[g], gpuHostX, sizeof(T) * gpuM * gpuK, cudaMemcpyHostToDevice));
    }
  }
  delete gpuHostX;
  std::cout << "Distributed X " << std::endl;
  return cudaSuccess;
}

template<typename T> cudaError_t FastKronHandle_gatherDistributedY(FastKronHandle& handle, T* dY[], T* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  //TODO: Make FastKronError type
  if (!handle.isDistributed_) return cudaErrorInvalidValue;
  //TODO: Check that hY is on host memory
  uint gpuM, gpuYCols, YCols;
  YCols = getYColumns(M, K, NumKronMats, KronMatCols, KronMatRows);
  handle.getDistributedSizes(M, YCols, gpuM, gpuYCols);
  T* gpuHostY = new T[gpuM * gpuYCols];
  std::cout << "Gather Y from all GPUs"<<std::endl;

  for(int gr = 0; gr < handle.gpusInM_; gr++) {
    for (uint gc = 0; gc < handle.gpusInK_; gc++) {
      uint g = gr * handle.gpusInK_ + gc;
      CUDA_CHECK(cudaSetDevice(g));
      //TODO: check that dX[g] is on GPU g
      CUDA_CHECK(cudaMemcpy(gpuHostY, dY[g], 
                            sizeof(T) * gpuM * gpuYCols,
                            cudaMemcpyDeviceToHost));
      const uint startGpuM = gpuM * gr;
      const uint startGpuN = gpuYCols * gc;
      for (int m = 0; m < gpuM; m++) {
        std::memcpy(&hY[(startGpuM+m)*YCols + startGpuN],
                    &gpuHostY[m * gpuYCols], sizeof(T)*gpuYCols);
      }
    }
  }
  
  delete gpuHostY;

  std::cout << "Gathered Y" << std::endl;

  return cudaSuccess;
}

template<> cudaError_t FastKronHandle::allocDistributedX(float* dX[], float* hX, uint M, uint K) {
  return FastKronHandle_allocDistributedX<float>(*this, dX, hX, M, K);
}

template<> cudaError_t FastKronHandle::allocDistributedX(double* dX[], double* hX, uint M, uint K) {
  return FastKronHandle_allocDistributedX<double>(*this, dX, hX, M, K);
}

template<> cudaError_t FastKronHandle::allocDistributedX(int* dX[], int* hX, uint M, uint K) {
  return FastKronHandle_allocDistributedX<int>(*this, dX, hX, M, K);
}

template<> cudaError_t FastKronHandle::gatherDistributedY(float* dY[], float* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  return FastKronHandle_gatherDistributedY<float>(*this, dY, hY, M, K, NumKronMats, KronMatCols, KronMatRows);
}

template<> cudaError_t FastKronHandle::gatherDistributedY(double* dY[], double* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  return FastKronHandle_gatherDistributedY<double>(*this, dY, hY, M, K, NumKronMats, KronMatCols, KronMatRows);
}

template<> cudaError_t FastKronHandle::gatherDistributedY(int* dY[], int* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  return FastKronHandle_gatherDistributedY<int>(*this, dY, hY, M, K, NumKronMats, KronMatCols, KronMatRows);
}

template<typename T> void FastKronHandle_init(FastKronHandle& handle, 
                                              int gpus, int gpusInM, int gpusInK, int gpuKrons) {
  //TODO: Support both modes. Single Process multi gpu and multi process multi gpu
  handle.isDistributed_ = gpus > 1;
  if (handle.isDistributed_) {
    //TODO: Setting DistComm in another function
    handle.setUseFusion(false);
    handle.numGPUs_ = gpus;
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

    handle.distComm_ = env::getDistComm();

    if (handle.distComm_ == DistComm::P2P) {
      if (!allP2PAccess) {
        std::cout << "P2P Access among GPUs not available using NCCL" << std::endl;
        handle.distComm_ = DistComm::DistCommNone;
      }
    } else if (handle.distComm_ == DistComm::NCCL) {
      int devs[gpus];
      handle.distComm_ = DistComm::NCCL;
      ncclUniqueId ncclId;
      ncclGetUniqueId(&ncclId);
      std::cout << "Initializing NCCL"<<std::endl;
      for (int i = 0; i < gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        handle.ncclComms.push_back(nullptr);
        devs[i] = i;
      }
      NCCLCHECK(ncclCommInitAll(&handle.ncclComms[0], gpus, devs));
    }

    if (handle.distComm_ == DistComm::DistCommNone) {
      if (allP2PAccess) {
        handle.distComm_ = DistComm::P2P;
      } else {
        int devs[gpus];
        handle.distComm_ = DistComm::NCCL;
        ncclUniqueId ncclId;
        ncclGetUniqueId(&ncclId);
        std::cout << "Initializing NCCL"<<std::endl;
        for (int i = 0; i < gpus; i++) {
          CUDA_CHECK(cudaSetDevice(i));
          handle.ncclComms.push_back(nullptr);
          devs[i] = i;
        }
        NCCLCHECK(ncclCommInitAll(&handle.ncclComms[0], gpus, devs));
      }
    }

    std::cout << "Using " << handle.distComm_ << " for distributed comm" << std::endl;

    if (gpusInK >= 1)
      handle.gpusInK_ = gpusInK;
    else
      handle.gpusInK_ = 2;//ilog2(gpus);
    
    if (gpusInM >= 1)
      handle.gpusInM_ = gpusInM;  
    else
      handle.gpusInM_ = 1;//ilog2(gpus);
      
    //TODO: Check that gpuKrons batch is valid, i.e., P1*P2..PBatch <= gpusInK
    if (gpuKrons > 0)
      handle.perGPUKronBatch_ = gpuKrons;
    else 
      handle.perGPUKronBatch_ = 1;

    //TODO: Check if gpusInK_ == 1 then perGPUKronBatch = NumKrons

    std::cout << "gpusInRows " << handle.gpusInM_ <<
                 " gpusInCols " << handle.gpusInK_ << 
                 " gpuKronBatch " << handle.perGPUKronBatch_ <<
                 std::endl;
    if (handle.gpusInK_ * handle.gpusInM_ != handle.numGPUs_)  {
      std::cout << "gpusInCols * gpusInRows != total gpus (" << 
                   handle.gpusInK_ * handle.gpusInM_ << "!= " << 
                   handle.numGPUs_<< ")" << std::endl;
      abort();
    }
    //TODO: Check that localKrons <= log (gpuK_)_P
    // handle.gpuM_ = handle.M_/handle.gpusInM_;
    // handle.gpuK_ = handle.K_/handle.gpusInK_;
    // handle.gpuN_ = handle.N_/handle.gpusInK_;
    
    //All gpus with same row shares the same barrier
    //TODO: free
    handle.barriers_ = new pthread_barrier_t[handle.gpusInM_];
    handle.threads_ = new thread_pool<ThreadArgs*>(handle.numGPUs_);

    for (int i = 0; i < handle.gpusInM_; i++) {
      int s = pthread_barrier_init(&handle.barriers_[i], NULL, handle.gpusInK_);
      //TODO: Create PTHREAD_CHECK?
      assert (s == 0);
    }
    
    // size_t tempN = handle.gpuK_;
    // size_t maxTempN = tempN;
    // for (int i = 0; i < handle.NumKronMats_; i++) {
    //   tempN = (tempN/handle.KronMatRows_[i])*handle.KronMatCols_[i];
    //   if (maxTempN < tempN)
    //     maxTempN = tempN;
    // }

    // size_t sz = handle.gpuM_ * maxTempN * sizeof(T);
    // std::cout << "Allocating temporaries of size "<< sz << std::endl;
    // std::cout << "Allocated temporaries"<<std::endl;

  } else {
    // size_t tempN = handle.K_;
    // size_t maxTempN = tempN;
    // for (int i = 0; i < handle.NumKronMats_; i++) {
    //   tempN = (tempN/handle.KronMatRows_[i])*handle.KronMatCols_[i];
    //   if (maxTempN < tempN)
    //     maxTempN = tempN;
    // }
    // size_t sz = handle.M_ * maxTempN * sizeof(T);
    // CUDA_CHECK(cudaMalloc(&handle.temp1_, sz));
    // CUDA_CHECK(cudaMalloc(&handle.temp2_, sz));
    // CUDA_CHECK(cudaMemset(handle.temp1_, 0, sz));
    // CUDA_CHECK(cudaMemset(handle.temp2_, 0, sz));
  }

  //Initialize compiledKernels map

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
        std::cout << kernel << std::endl;
      }
      numKernels += iter.second.size();
    }
    std::cout << "Number of kernels loaded: " << numKernels << std::endl;
  }
}

template<> void FastKronHandle::init<float>(int gpus, int gpusInM, int gpusInK, int gpuLocalKrons) {
  FastKronHandle_init<float>(*this, gpus, gpusInM, gpusInK, gpuLocalKrons);
}

template<> void FastKronHandle::init<int>(int gpus, int gpusInM, int gpusInK, int gpuLocalKrons) {
  FastKronHandle_init<int>(*this, gpus, gpusInM, gpusInK, gpuLocalKrons);
}

template<> void FastKronHandle::init<double>(int gpus, int gpusInM, int gpusInK, int gpuLocalKrons) {
  FastKronHandle_init<double>(*this, gpus, gpusInM, gpusInK, gpuLocalKrons);
}

void FastKronHandle::free() {
  if (isDistributed_) {
    //TODO: Clear everything
    for (uint g = 0; g < numGPUs_; g++) {
      // CUDA_CHECK(cudaFree(gpuTemp1_[g]));
      // CUDA_CHECK(cudaFree(gpuTemp2_[g]));
    }

    // delete[] gpuTemp1_;
    // delete[] gpuTemp2_;

    // gpuTemp1_ = nullptr;
    // gpuTemp2_ = nullptr;
  } else {
    // CUDA_CHECK(cudaFree(temp1_));
    // CUDA_CHECK(cudaFree(temp2_));
  
    // temp1_ = nullptr;
    // temp2_ = nullptr;  
  }
}