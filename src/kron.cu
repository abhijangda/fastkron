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

#define CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

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
#define DIVUP(x, y) (((x) + (y) - 1)/((y)))

#define C_IN_REG
#define EVAL

//utils.h
static constexpr int log2(uint n) {return 31 - __builtin_clz(n);}
static constexpr int log2(int n) {return 31 - __builtin_clz(n);}

enum RowParallelismTy {
  Low = 0,
  Medium,
  High,
  Num = 3,
};

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
  int kEqVar = (maxColsAShape.ColsA == shape.ColsA) ? 1 : 0;
  auto iter = compiledKernels.find(maxColsAShape);
  if (iter == compiledKernels.end()) {
    std::cout << "No kernel found" << std::endl;
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
  return KernelInfo();
}

//Launch cuda kernels
template<typename T, uint NumFusedKerns>
cudaError_t generalSlicedMatmul(FastKronHandle& handle, KernelInfo& kernelInfo, const uint kronIndex, T* x, T* kronMat[NumFusedKerns], 
                                T* kronGemmResult,
                                const uint M, const uint N, const uint K, 
                                const uint KronMatCols[NumFusedKerns], const uint KronMatRows[NumFusedKerns],
                                const uint gr, const uint gc, const uint LocalKrons, bool storeToDistMems, cudaStream_t stream) {
  cudaError_t status;
  RowParallelismTy rowParallelism = RowParallelismTy::Low;
  dim3 grid;
  dim3 block;

  const int KP_K_BATCH = 1;
  uint N_COARSE_TB = 1; //(M > 100) ? 2 : 1;
  uint max_k;
  uint min_k;
  uint max_k_kernel = 1;
  uint row_mod_tile_zero = 0;
  uint typeKernelIdx = typeKernelIndex((T)0);

  const uint NumThreads = kernelInfo.NumThreads;
  {
    const uint CRegRows = kernelInfo.CRegRows;
    const uint CRegCols = kernelInfo.CRegCols;
    const uint MaxColsC = kernelInfo.MaxColsA;
    const uint KronRows = kernelInfo.KronRows;
    uint c1 = MAX(1, NumThreads/((MaxColsC/kernelInfo.KronRows)/CRegRows));
    
    if (kernelInfo.TileKronCols != c1 * CRegCols) {
      printf("Invalid configuration: TileKronCols %d != c1*CRegCols %d; NumThreads %d CRegRows %d CRegCols %d MaxColsC %d\n", 
              kernelInfo.TileKronCols, c1 * CRegCols, NumThreads, CRegRows, CRegCols, MaxColsC);
      abort();
    }
    if (MaxColsC/KronRows > kernelInfo.NumThreads*c1* kernelInfo.CRegRows) {
      printf("MaxColsC/KronRows %d kernelInfo.NumThreads*c1* kernelInfo.CRegRows %d\n", MaxColsC/KronRows, kernelInfo.NumThreads*c1* kernelInfo.CRegRows);
      printf("Invalid configuration: MaxColsC %d KronRows %d NumThreads %d CRegRows %d CRegCols %d\n",
              MaxColsC, KronRows, NumThreads, CRegRows, CRegCols);
      abort();
    }
  }
  //Create the grid and thread block
  grid = {
            (K/kernelInfo.MaxColsA) * DIVUP(KronMatCols[0], kernelInfo.TileKronCols),
            DIVUP(M, kernelInfo.TileRowsA),
            1// DIVUP(KronMatRows[kronMat], EXTERNAL_KP_K_TILE_)
          };
  block = {
            NumThreads, 
            1, 
            1
          };

  KernelParams<T, NumFusedKerns> params (M, N, K, 
                                         KronMatRows, 
                                         KronMatCols, x, 
                                         kronMat, 
                                         kronGemmResult, 
                                         kronIndex);
  auto ttt = (LocalKrons == 3) ? (T**)handle.gpuTemp1_ : (T**)handle.gpuTemp2_;
  DistributedParams<T> distParams(ttt[0], ttt[1], gr, gc, handle.numGPUs_, handle.K_, handle.N_, LocalKrons, storeToDistMems);

  typedef void (*KronMatmulKernel)(KernelParams<T, NumFusedKerns>, DistributedParams<T>);
  //Create kernel args;
  // void *args[] = {(void*)&params};
  ((KronMatmulKernel)kernelInfo.kernel)<<<grid, block,0,stream>>>(params, distParams);
  status = cudaGetLastError();
  // status = cudaLaunchKernel((const void*)cuda_gemm_func, grid, block, &args[0], 0, stream);
  // if (status != cudaSuccess)
  //   return status;
  return status;
}

//TODO: These methods that take handle should be private methods of FastKronHandle
TunedKernelsSeries selectKernelSeries(FastKronHandle& handle, const uint NumKronMats,
                                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[]) {
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

    auto selectedKernel = selectKernel(KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                       prevTempN, M, NumFusedKerns});
    tunedSeries.push_back({selectedKernel, kronMat - NumFusedKerns, kronMat, prevTempN, 0.0f});
    prevTempN = currTempN;
  }

  return tunedSeries;
}

template<typename T, typename VecT>
cudaError_t singleGPUKronMatmul(FastKronHandle& handle, const uint NumKronMats, T* x, T* kronMats[], 
                                T** result,
                                uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], 
                                cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  if (result == NULL) return cudaErrorInvalidValue;
  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return cudaErrorInvalidValue;
  T* kronGemmResults[2] = {(T*)handle.temp_, (T*)handle.result_};
  T* prevKronResult = x;
  T* currKronResult = kronGemmResults[0];
  //TODO: Assumes all factors are of same size and square shape
  TunedKernelsSeries kernelSeries;
  if (handle.tunedKernelSeries.size() > 0) {
    kernelSeries = handle.tunedKernelSeries;
  } else {
    kernelSeries = selectKernelSeries(handle, NumKronMats, M, N, K, 
                                      KronMatCols, KronMatRows);
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
      currTempN = (prevTempN/FusedKronMatRows[k])*FusedKronMatCols[k];
    }
    cudaError_t status;

    KernelInfo selectedKernel = kernel.kernel;
/*
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
    */
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

  *result = prevKronResult;

  return cudaSuccess;
}

// template<typename T>
// struct LinkedListNode {
//   T data;
//   std::vector<struct TreeNode> children;

//   TreeNode(T& data_) : data(data_) {}
//   void addChild(struct TreeNode child) {
//     children.push_back(child);
//   }
// }

float minExecTimeOfSeries(uint M, uint K, const uint NumKronMats, 
                          uint KronMatCols[], uint KronMatRows[],
                          uint startKron,
                          TunedKernelsSeries& tunedKernels,
                          std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>> bestKernels) {
  if (startKron >= NumKronMats) return 0;

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
                                            tempK, M, NumFusedKerns};

    if (bestKernels.find(shape) == bestKernels.end()) continue;
    auto iter = bestKernels.find(shape);
    TunedKernelsSeries epilogueKernels;
    float kernelTime = iter->second.second;
    float epilogueTime = minExecTimeOfSeries(M, K, NumKronMats, KronMatCols,
                                             KronMatRows, endKron + 1, epilogueKernels, 
                                             bestKernels);
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
cudaError_t autotune(FastKronHandle& handle, const uint NumKronMats, T* x, T* kronMats[], 
                     uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                     cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  std::cout << "N " << N << " K " << K << " KronMatCols[0] " << KronMatCols[0] << " KronMatRows[0] " << KronMatRows[0] << std::endl;
  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return cudaErrorInvalidValue;
  T* kronGemmResults[2] = {(T*)handle.temp_, (T*)handle.result_};
  //For performance eval we do not need these to contain any value
  T* prevKronResult = kronGemmResults[0];
  T* currKronResult = kronGemmResults[1];
  std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>> bestKernels;
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
    //Include KronMats [startKron, ..., endKron]
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
    cudaError_t status;
    KronMatmulShape shape = KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                            tempN, M, NumFusedKerns};
    if (bestKernels.find(shape) != bestKernels.end()) {
      continue;
    }
    KernelInfo bestKernel;
    float minTime = std::numeric_limits<float>::max();
    const uint runs = 10;
    for (auto shapeAndKernels : compiledKernels) {
      if (!shapeAndKernels.first.sameKronSize(shape))
        continue;
      for (auto kernel : shapeAndKernels.second) {
        if (!kernel.canCompute(shape, NumFusedKerns)) continue;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int r = 0; r < 5 + runs; r++) {
          if (r == 5) CUDA_CHECK(cudaEventRecord(start, stream));
          /*
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
          }*/
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
  std::cout << "Finding min execution time of the series" << std::endl;
  TunedKernelsSeries tunedKernels;
  float minTime = minExecTimeOfSeries(M, K, NumKronMats,
                                      KronMatCols, KronMatRows, 0,
                                      tunedKernels, bestKernels);
  std::cout <<"Minimum Time " << minTime << " through kernels: " << std::endl;
  for (auto iter = tunedKernels.rbegin(); iter != tunedKernels.rend(); iter++) {
    std::cout << "  " << (*iter) << std::endl;
  }
  handle.tunedKernelSeries = tunedKernels;
  return cudaSuccess;
}

cudaError_t kronSGEMMTune(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], 
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

template<typename T, typename VecT>
void perGPUKronMatmul(ThreadArgs* thArgs) {
  // ThreadArgs<T>& thArgs = *(ThreadArgs<T>*)arg;

  FastKronHandle& handle = *thArgs->handle;
  uint NumKronMats = thArgs->NumKronMats;
  T* x = (T*)thArgs->x;
  T** kronMats = (T**)thArgs->kronMats;
  T* result = (T*)thArgs->result;
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

  uint g = gr * gpusInK_ + gc;
  CUDA_CHECK(cudaSetDevice(g));

  cudaError_t status;
  
  //Temporaries are swaped after every slicedMatmul
  //TODO: User supplied result should be used as a temp and the final results are written in it
  //TODO: What if Rows are not multiple of GPUs in Rows
  T* innerResults[2] = {(T*)handle.gpuTemp1_[g], (T*)handle.gpuTemp2_[g]};
  // std::cout << "handle.gpuM_ " << handle.gpuM_ << " handle.gpuK_ " <<handle.gpuK_ << " gpusInCols " << gpusInCols << " gpusInRows " << gpusInRows << " K " << K << std::endl;
  if (gc == 0) std::cout <<"g " << g<< " innerResults = " << "{" << innerResults[0] << ", " << innerResults[1] << "}" << std::endl;
  T* innerPrevResult;
  T* innerCurrResult;
  
  uint startGpuM = handle.gpuM_ * gr;
  const uint gpuM = min(handle.gpuM_, M - startGpuM);
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
    // std::cout << "io " << io << " gr " << gr << " gc " << gc << "g " << g <<  std::endl;
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
      TunedKernelsSeries kernelSeries;
      kernelSeries = selectKernelSeries(handle, KronMulBatchSize, gpuM, handle.gpuK_, handle.gpuK_, 
                                        KronMatCols, KronMatRows);
      uint slicedMuls = 0;
      for (auto kernel : kernelSeries) {
        //TODO: probably will need to change for fused kernels
        uint kronMat = kernel.end + (NumKronMats - io - KronMulBatchSize);
        T* krons[] = {kronMats[g * NumKronMats + kronMat]}; 
        uint kronCols[1] = {KronMatCols[kronMat]};
        uint kronRows[1] = {KronMatRows[kronMat]};
        // if (gc == 0) std::cout << "671: " << (slicedMuls == KronMulBatchSize - 1 and KronMulBatchSize > 1) << std::endl;
        // if (slicedMuls == KronMulBatchSize - 1) {
        //   {
        //     int s = pthread_barrier_wait(thArgs->barrier);
        //     assert (s == 0 || s == PTHREAD_BARRIER_SERIAL_THREAD);
        //   }
        // }
        cudaError_t status = generalSlicedMatmul<T, 1>(handle, kernel.kernel, kronMat, innerPrevResult, 
            krons, innerCurrResult, gpuM, handle.gpuK_, handle.gpuK_, 
            kronCols, kronRows, gr, gc, KronMulBatchSize, 
            (slicedMuls == KronMulBatchSize - 1) ? false : false, stream[g]);
        
        CUDA_CHECK(cudaStreamSynchronize(stream[g]));
        // if (gc == 0) printf("slicedMuls %d innerCurrResult %p innerPrevResult %p\n", slicedMuls, innerCurrResult, innerPrevResult);
        if (status != cudaSuccess) goto end;

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
      
      {
        int s = pthread_barrier_wait(thArgs->barrier);
        assert (s == 0 || s == PTHREAD_BARRIER_SERIAL_THREAD);
      }

      // {
      //   CUDA_CHECK(cudaSetDevice(0));
      //   CUDA_CHECK(cudaDeviceSynchronize());
      //   CUDA_CHECK(cudaSetDevice(1));
      //   CUDA_CHECK(cudaDeviceSynchronize());
      //   CUDA_CHECK(cudaSetDevice(g));
      // }

      std::cout << "g " << g << " innerPrevResult "<< innerPrevResult << std::endl;
      auto ttt = (KronMulBatchSize == 3) ? (T**)handle.gpuTemp2_ : (T**)handle.gpuTemp2_;
      const uint PerGPUK = 64U*64U*64U*64U/2U;
      dim3 grid = {handle.gpuK_/8192, gpuM, 1};
      dim3 block = {256, 1, 1};
      copyToGPUsInK<T, float2, 2U, PerGPUK, 8192, 256><<<grid,block,0,stream[g]>>>
                                                       (gpuM, handle.N_, handle.K_, innerPrevResult, 
                                                        ttt[0], ttt[1], gr, gc, KronMulBatchSize);
    
      CUDA_CHECK(cudaStreamSynchronize(stream[g]));
      // {
      //   int s = pthread_barrier_wait(thArgs->barrier);
      //   assert (s == 0 || s == PTHREAD_BARRIER_SERIAL_THREAD);
      // }
      
      // {
      //   CUDA_CHECK(cudaSetDevice(0));
      //   CUDA_CHECK(cudaDeviceSynchronize());
      //   CUDA_CHECK(cudaSetDevice(1));
      //   CUDA_CHECK(cudaDeviceSynchronize());
      //   CUDA_CHECK(cudaSetDevice(g));
      // }

      if (gc == 0) {
        innerCurrResult = innerPrevResult;
        innerPrevResult = ttt[0];
      } else {
        innerCurrResult = innerPrevResult;
        innerPrevResult = ttt[1];
      }
      // if (innerPrevResult == innerResults[0]) {        
      //   innerCurrResult = innerResults[1];
      // } else if (innerPrevResult == innerResults[1]) {
      //   innerCurrResult = innerResults[0];
      // }

      // CUDA_CHECK(cudaSetDevice(0));
      // CUDA_CHECK(cudaDeviceSynchronize());
      // CUDA_CHECK(cudaSetDevice(1));
      // CUDA_CHECK(cudaDeviceSynchronize());
      // CUDA_CHECK(cudaSetDevice(g));

      // if (g == 0 && MaxI == 3) {
      //   printf("683\n");
      //   printGPUArray<float>(handle.gpuM_, handle.gpuK_, 64*64*64, innerPrevResult, stream[g]);
      // }
      // printf("684\n");

      if (false && handle.gpusInK_ > 1) {
        //Call we want to use NCCL Send/Recv
        {
          const uint SliceRows = handle.gpuM_;
          const uint SliceCols = handle.gpuK_/handle.gpusInK_;
          const size_t sendRecvSize = SliceRows * SliceCols;
          const uint startRow = 0;
          const uint startCol = gc * SliceCols;
          matrixSlice(handle.gpuM_, handle.gpuK_, innerPrevResult, 
                        startRow, startCol, SliceRows, SliceCols,
                        (float*)handle.recvTemps_[g], stream[g], g, io, true);
          dim3 grid = {handle.gpuM_, 1,1};
          dim3 block = {256, 1, 1};
          storeGPUTile<T, VecT, 256><<<grid, block, 0, stream[g]>>>(M, N, K, KronMatRows[0], KronMatRows[0], gc, handle.gpusInK_, 
                                                                    (float*)handle.recvTemps_[g], handle.gpuM_, handle.gpuK_,
                                                                    innerCurrResult, gc, KronMulBatchSize, io, (io == 0 and g == 1));
          CUDA_CHECK(cudaStreamSynchronize(stream[g]));
        }

        //All GPUs with the same gr share their intermediates
        for (int dst = 0; dst < handle.gpusInK_; dst++) {
          const uint SliceRows = handle.gpuM_;
          const uint SliceCols = handle.gpuK_/handle.gpusInK_;
          const size_t sendRecvSize = SliceRows * SliceCols;
          if (dst == gc) {
            for (int src = 0; src < handle.gpusInK_; src++) {
              // printf("g %d dst %d src %d\n", g, dst, src);
              if (src == dst) {
                // const uint startRow = 0;
                // const uint startCol = dst * SliceCols;
                // matrixSlice(handle.gpuM_, handle.gpuK_, innerPrevResult, 
                //             startRow, startCol, SliceRows, SliceCols,
                //             (float*)handle.recvTemps_[g], stream[g], g, io, true);
                // if (g == 0 and io == 1) printGPUArray<float>(handle.gpuM_, 1024, (float*)handle.recvTemps_[g], stream[g]);

              } else {
                NCCLCHECK(ncclRecv(handle.recvTemps_[g], sendRecvSize, ncclFloat, gr * handle.gpusInK_ + src, handle.ncclComms[g], stream[g]));
                CUDA_CHECK(cudaStreamSynchronize(stream[g]));
              // if (g == 0) {
              //   printf("704\n");
              //   printGPUArray<float>(handle.gpuM_, handle.gpuK_, (float*)handle.recvTemps_[g], stream[g]);
              //   printf("699 src %d dst %d g %d\n", src, dst, g);
              // printf("713\n");
              // printGPUArray<float>(handle.gpuM_, handle.gpuK_, (float*)handle.recvTemps_[g], stream[g]);
              // printf("715\n");
              dim3 grid = {handle.gpuM_, 1,1};
              dim3 block = {256, 1, 1};
              storeGPUTile<T, VecT, 256><<<grid, block, 0, stream[g]>>>(M, N, K, KronMatRows[0], KronMatRows[0], dst, handle.gpusInK_, 
                                                                        (float*)handle.recvTemps_[g], handle.gpuM_, handle.gpuK_,
                                                                        innerCurrResult, src, KronMulBatchSize, io, (io == 0 and g == 1));
              CUDA_CHECK(cudaStreamSynchronize(stream[g]));
              // if (io == 0 and g == 1) printGPUArray<float>(80, 2048, ((io == 0) ? 16.0 :256.0f), (const float*)innerCurrResult, stream[g]);
              // }
              }
            }
          } else {
            const uint startRow = 0;
            const uint startCol = dst * SliceCols;
            matrixSlice(handle.gpuM_, handle.gpuK_, innerPrevResult, 
                        startRow, startCol, SliceRows, SliceCols,
                        (float*)handle.sendTemps_[g], stream[g], g, io);
            CUDA_CHECK(cudaStreamSynchronize(stream[g]));
            // if (g == 1 && dst == 0) {
            //    printGPUArray<float>(SliceRows, SliceCols, (float*)handle.sendTemps_[g], stream[g]);
            //    printf("699 dst %d g %d\n", dst, g);
            // }
            NCCLCHECK(ncclSend(handle.sendTemps_[g], sendRecvSize, ncclFloat, gr * handle.gpusInK_ + dst, handle.ncclComms[g], stream[g]));
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
      // if (io == 0 and g == 1) printGPUArray<float>(80, 2048, ((io == 0) ? 16.0 :256.0f), (const float*)innerCurrResult, stream[g]);

      //  printf("737 io %d\n", io);
      //  if (g == 0) printGPUArray<float>(handle.gpuM_, handle.gpuK_, (float*)innerCurrResult, stream[g]);
            
      // printf("737 io %d\n", io);
      

      
      // return;
      //Copy uvaCurrResult to kronGemmResult
      // if (uvaColsX < K) {
      //   // CUDA_CHECK(cudaDeviceSynchronize());
      //   // printf("copyUVATempToY\n");
      //   dim3 grid = {outOfCoreRows, 1,1};
      //   dim3 block = {256, 1, 1};
      //   copyUVATempToY<T, VecT, 256><<<grid, block, 0, stream[g]>>>(M, N, K, KronMatRows[io], KronMatRows[io], innerCurrResult, outOfCoreRows, uvaColsX,
      //                                                               &outerCurrResult[startOutofCoreRows * K], uvaPart/uvaColsX, KronMulBatchSize, io);
      //   // CUDA_CHECK(cudaDeviceSynchronize());
      //   // printf("Done\n");
      // } else {
      //   if (innerPrevResult == innerResults[0]) {
      //     outerPrevResult = kronGemmResults[0];
      //     outerCurrResult = kronGemmResults[1];
      //   }
      //   else {
      //     outerPrevResult = kronGemmResults[1];
      //     outerCurrResult = kronGemmResults[0];
      //   }

      //   // printf("outerPrevResult %p outerCurrResult %p\n", outerPrevResult, outerCurrResult);
      // }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream[g]));
    printf("820: g %d io %d\n", g, io);
    // for (int gg = 0; gg < 2; gg++) {
    //   CUDA_CHECK(cudaSetDevice(gg));
    //   CUDA_CHECK(cudaDeviceSynchronize());
    // }
    // CUDA_CHECK(cudaSetDevice(g));
    {
      int s = pthread_barrier_wait(thArgs->barrier);
      assert (s == 0 || s == PTHREAD_BARRIER_SERIAL_THREAD);
    }
    printf("823: g %d thArgs->barrier %p io %d\n", g, thArgs->barrier, io);
    // if (io == 0) {
    //   // printf("683 io %d\n", io);
    //   printGPUArray<float>(handle.gpuM_, handle.gpuK_, 64*64*64, innerPrevResult, stream[g]);
    //   CUDA_CHECK(cudaStreamSynchronize(stream[g]));
    // }
  }

  end:
  thArgs->threadResult = {status, (void*)innerPrevResult};
}

template<typename T, typename VecT>
cudaError_t distributedKronMatmul(FastKronHandle& handle, const uint NumKronMats, T* x[], T* kronMats[], T* result[],
                                  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                                  cudaStream_t streams[]) {
  if (result == NULL)                       return cudaErrorInvalidValue;
  if (handle.gpuM_ > M)            return cudaErrorInvalidValue;
  if (NumKronMats < handle.perGPUKronBatch_) return cudaErrorInvalidValue;

  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return cudaErrorInvalidValue;
  
  std::cout << "Running distributed KronMatmul" << std::endl;
  const uint gpuM = handle.gpuM_;
  const uint gpuK = handle.gpuK_; //handle.OutofCoreKronBatch_ * power(KronMatRows[0], handle.OutofCoreKrons_); //KronMatCols[0] * KronMatCols[0]* KronMatCols[0]* KronMatCols[0] * KronMatCols[0] * KronMatCols[0];
  const uint batchedKronMuls = handle.perGPUKronBatch_;

  // printf("MaxInnerKrons %d uvaColsX %d K %d handle.outOfCoreTemp1_ %p\n", handle.OutofCoreKrons_, uvaColsX, K, handle.outOfCoreTemp1_);
  double timeStart = getCurrTime();

  thread_pool<ThreadArgs*>::task tasks[handle.numGPUs_];
  ThreadArgs threadArgs[handle.numGPUs_];

  for (uint thread = 0; thread < handle.numGPUs_; thread++) {
    ThreadArgs args = ThreadArgs(
      &handle,
      NumKronMats,
      (void*)x[thread],
      (void**)kronMats,
      (void*)result[thread],
      M,
      N,
      K,
      &KronMatCols[0],
      &KronMatRows[0],
      streams,
      thread/handle.gpusInK_,
      thread % handle.gpusInK_,
      handle.gpusInM_,
      handle.gpusInK_,
      &handle.barriers_[thread/handle.gpusInK_]
    );

    threadArgs[thread] = args;
    //TODO: make this ThreadArgs& instead of ThreadArgs*
    tasks[thread] = thread_pool<ThreadArgs*>::task(perGPUKronMatmul<T, VecT>, &threadArgs[thread]);
    // threads[thread] = std::thread(perGPUKronMatmul<T, VecT>, std::ref(threadArgs[thread]));
  }

  handle.threads_->execute_tasks(tasks);
  handle.threads_->join_tasks();

  cudaError_t status;
  for (uint thread = 0; thread < handle.numGPUs_; thread++) {
    status = threadArgs[thread].threadResult.status;
    result[thread] =(T*)threadArgs[thread].threadResult.result;
  }
  double timeEnd = getCurrTime();

  printf("531: time %lf microseconds\n", timeEnd - timeStart);
  // 
  return status;
}

/**************************************************
          Library Functions
***************************************************/
cudaError_t kronSGEMM(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream) {
  return singleGPUKronMatmul<float, float4>(handle, NumKronMats, x, kronMats, result, 
                                            M, N, K, KronMatCols, KronMatRows, stream);
}

cudaError_t kronIGEMM(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], int** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream) {
  return singleGPUKronMatmul<int, int4>(handle, NumKronMats, x, kronMats, result, 
                                            M, N, K, KronMatCols, KronMatRows, stream);
}

cudaError_t kronDGEMM(FastKronHandle& handle, const uint NumKronMats, double* x, double* kronMats[], double** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream) {
  return singleGPUKronMatmul<double, double4>(handle, NumKronMats, x, kronMats, result, 
      M, N, K, KronMatCols, KronMatRows, stream);
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
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t streams[]) {
  return distributedKronMatmul<float, float4>(handle, NumKronMats, x, kronMats, result, M, N, K, 
                                              KronMatCols, KronMatRows, streams);
}

template<typename T> cudaError_t FastKronHandle_allocDistributedX(FastKronHandle& handle, T* dX[], T* hX) {
  //TODO: Make FastKronError type
  if (!handle.isDistributed_) return cudaErrorInvalidValue;
  //TODO: Check that hX is on host memory
  T* gpuHostX = new T[handle.gpuM_ * handle.gpuK_];
  std::cout << "Distributing X to all GPUs"<<std::endl;
  for (int g = 0; g < handle.numGPUs_; g++) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMalloc(&dX[g], sizeof(T) * handle.gpuM_ * handle.gpuK_));
  }

  for(int gr = 0; gr < handle.gpusInM_; gr++) {
    for (uint gc = 0; gc < handle.gpusInK_; gc++) {
      const uint g = gr * handle.gpusInK_ + gc;
      CUDA_CHECK(cudaSetDevice(g));
      uint startGpuM = handle.gpuM_ * gr;
      uint startGpuK = handle.gpuK_ * gc;
        
      for (uint m = 0; m < handle.gpuM_; m++) {
        std::memcpy(&gpuHostX[m * handle.gpuK_], &hX[(startGpuM+m)*handle.K_ + startGpuK], sizeof(T)*handle.gpuK_);
      }
      CUDA_CHECK(cudaMemcpy(dX[g], gpuHostX, sizeof(T) * handle.gpuM_ * handle.gpuK_, cudaMemcpyHostToDevice));
    }
  }
  delete gpuHostX;
  std::cout << "Distributed X " << std::endl;
  return cudaSuccess;
}

template<typename T> cudaError_t FastKronHandle_gatherDistributedY(FastKronHandle& handle, T* dY[], T* hY) {
  //TODO: Make FastKronError type
  if (!handle.isDistributed_) return cudaErrorInvalidValue;
  //TODO: Check that hY is on host memory

  T* gpuHostY = new T[handle.gpuM_ * handle.gpuK_];
  std::cout << "Gather Y from all GPUs"<<std::endl;

  for(int gr = 0; gr < handle.gpusInM_; gr++) {
    for (uint gc = 0; gc < handle.gpusInK_; gc++) {
      uint g = gr * handle.gpusInK_ + gc;
      CUDA_CHECK(cudaSetDevice(g));
      //TODO: check that dX[g] is on GPU g
      CUDA_CHECK(cudaMemcpy(gpuHostY, dY[g], 
                            sizeof(T) * handle.gpuM_ * handle.gpuK_, 
                            cudaMemcpyDeviceToHost));
      const uint startGpuM = handle.gpuM_ * gr;
      const uint startGpuK = handle.gpuK_ * gc;
      for (int m = 0; m < handle.gpuM_; m++) {
        std::memcpy(&hY[(startGpuM+m)*handle.K_ + startGpuK],
                    &gpuHostY[m * handle.gpuK_], sizeof(T)*handle.gpuK_);
      }
    }
  }
  
  delete gpuHostY;

  std::cout << "Gathered Y" << std::endl;

  return cudaSuccess;
}

template<> cudaError_t FastKronHandle::allocDistributedX(float* dX[], float* hX) {
  return FastKronHandle_allocDistributedX<float>(*this, dX, hX);
}

template<> cudaError_t FastKronHandle::allocDistributedX(double* dX[], double* hX) {
  return FastKronHandle_allocDistributedX<double>(*this, dX, hX);
}

template<> cudaError_t FastKronHandle::allocDistributedX(int* dX[], int* hX) {
  return FastKronHandle_allocDistributedX<int>(*this, dX, hX);
}

template<> cudaError_t FastKronHandle::gatherDistributedY(float* dY[], float* hY) {
  return FastKronHandle_gatherDistributedY<float>(*this, dY, hY);
}

template<> cudaError_t FastKronHandle::gatherDistributedY(double* dY[], double* hY) {
  return FastKronHandle_gatherDistributedY<double>(*this, dY, hY);
}

template<> cudaError_t FastKronHandle::gatherDistributedY(int* dY[], int* hY) {
  return FastKronHandle_gatherDistributedY<int>(*this, dY, hY);
}

template<typename T> void FastKronHandle_init(FastKronHandle& handle, bool isDistributed, 
                                              int gpus, int gpusInM, int gpusInK, int gpuKrons) {
  //TODO: Support both modes. Single Process multi gpu and multi process multi gpu
  handle.isDistributed_ = isDistributed;
  if (isDistributed) {
    handle.numGPUs_ = gpus;
    int devs[gpus];
    // ncclUniqueId ncclId;
    // ncclGetUniqueId(&ncclId);
    // std::cout << "Initializing NCCL"<<std::endl;
    // for (int i = 0; i < gpus; i++) {
    //   CUDA_CHECK(cudaSetDevice(i));
    //   handle.ncclComms.push_back(nullptr);
    //   devs[i] = i;
    //   //TODO: clear ncclComms
    //   // NCCLCHECK(ncclCommInitRank(&handle.ncclComms[i], gpus, ncclId, i));
    // }
    // NCCLCHECK(ncclCommInitAll(&handle.ncclComms[0], gpus, devs));
    // std::cout << "Initialized NCCL"<<std::endl;
    // assert (gpus == 4);
    for (int g1 = 0; g1 < gpus; g1++) {
      for (int g2 = 0; g2 < gpus; g2++) {
        if (g1 == g2) continue;
        CUDA_CHECK(cudaSetDevice(g1));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(g2, 0));
      }
    }
    if (gpusInK >= 1)
      handle.gpusInK_ = gpusInK;
    else
      handle.gpusInK_ = 2;//ilog2(gpus);
    
    if (gpusInM >= 1)
      handle.gpusInM_ = gpusInM;  
    else
      handle.gpusInM_ = 1;//ilog2(gpus);
    
    if (gpuKrons > 0)
      handle.perGPUKronBatch_ = gpuKrons;
    else 
      handle.perGPUKronBatch_ = 1;

    //TODO: Check if gpusInK_ == 1 then perGPUKronBatch = NumKrons

    std::cout << "gpusInRows " << handle.gpusInM_ <<
                 " gpusInCols " << handle.gpusInK_ << 
                 " gpuKronBatch" << handle.perGPUKronBatch_ <<
                 std::endl;
    if (handle.gpusInK_ * handle.gpusInM_ != handle.numGPUs_)  {
      std::cout << "gpusInCols * gpusInRows != total gpus (" << 
                   handle.gpusInK_ * handle.gpusInM_ << "!= " << 
                   handle.numGPUs_<< ")" << std::endl;
      abort();
    }
    //TODO: Check that localKrons <= log (gpuK_)_P
    handle.gpuM_ = handle.M_/handle.gpusInM_;
    handle.gpuK_ = handle.K_/handle.gpusInK_;
    handle.gpuTemp1_ = new void*[gpus];
    handle.gpuTemp2_ = new void*[gpus];
    handle.sendTemps_ = new void*[gpus];
    handle.recvTemps_ = new void*[gpus];
    
    //All gpus with same row shares the same barrier
    //TODO: free
    handle.barriers_ = new pthread_barrier_t[handle.gpusInM_];
    handle.threads_ = new thread_pool<ThreadArgs*>(handle.numGPUs_);

    for (int i = 0; i < handle.gpusInM_; i++) {
      int s = pthread_barrier_init(&handle.barriers_[i], NULL, handle.gpusInK_);
      //TODO: Create PTHREAD_CHECK?
      assert (s == 0);
    }
    std::cout << "Allocating temporaries"<<std::endl;
    size_t sz = handle.gpuM_ * handle.gpuK_ * sizeof(T);
    std::cout << "sz " << sz << std::endl;
    for (int g = 0; g < gpus; g++) {
      CUDA_CHECK(cudaSetDevice(g));
      CUDA_CHECK(cudaMalloc(&handle.gpuTemp1_[g], sz));
      CUDA_CHECK(cudaMalloc(&handle.gpuTemp2_[g], sz));
      CUDA_CHECK(cudaMemset(handle.gpuTemp1_[g], 0, sz));
      CUDA_CHECK(cudaMemset(handle.gpuTemp2_[g], 0, sz));
      //TODO: Figure this size
      CUDA_CHECK(cudaMalloc(&handle.sendTemps_[g], sz));
      CUDA_CHECK(cudaMalloc(&handle.recvTemps_[g], sz));
    }
    std::cout << "Allocated temporaries"<<std::endl;

  } else {
    size_t tempN = handle.K_;
    size_t maxTempN = tempN;
    for (int i = 0; i < handle.NumKronMats_; i++) {
      tempN = (tempN/handle.KronMatRows_[i])*handle.KronMatCols_[i];
      if (maxTempN < tempN)
        maxTempN = tempN;
    }

    size_t sz = handle.M_ * maxTempN * sizeof(T);
    CUDA_CHECK(cudaMalloc(&handle.temp_, sz));
    CUDA_CHECK(cudaMalloc(&handle.result_, sz));
    CUDA_CHECK(cudaMemset(handle.temp_, 0, sz));
  }

  //Initialize compiledKernels map

  for (uint i = 0; i < sizeof(KronGemmKernels)/sizeof(KernelInfo); i++) {
    KernelInfo& info = KronGemmKernels[i];
    KronMatmulShape shape {info.KronCols, info.KronRows, info.MaxColsA, 0, info.NumFusedKerns};
    auto iter = compiledKernels.find(shape);
    if (iter == compiledKernels.end()) {
      compiledKernels.emplace(std::make_pair(shape, std::vector<KernelInfo>()));
    }
    compiledKernels.at(shape).push_back(info);
  }
  
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

template<> void FastKronHandle::init<float>() {
  FastKronHandle_init<float>(*this, false, 1, 1, 1, 1);
}

template<> void FastKronHandle::init<int>() {
  FastKronHandle_init<int>(*this, false, 1, 1, 1, 1);
}

template<> void FastKronHandle::init<double>() {
  FastKronHandle_init<double>(*this, false, 1, 1, 1, 1);
}

template<> void FastKronHandle::initDistributed<float>(int gpus, int gpusInM, int gpusInK, int localKrons) {
  FastKronHandle_init<float>(*this, true, gpus, gpusInM, gpusInK, localKrons);
}

template<> void FastKronHandle::initDistributed<int>(int gpus, int gpusInM, int gpusInK, int localKrons) {
  FastKronHandle_init<int>(*this, true, gpus, gpusInM, gpusInK, localKrons);
}

template<> void FastKronHandle::initDistributed<double>(int gpus, int gpusInM, int gpusInK, int localKrons) {
  FastKronHandle_init<double>(*this, true, gpus, gpusInM, gpusInK, localKrons);
}

void FastKronHandle::free() {
  if (isDistributed_) {
    for (uint g = 0; g < numGPUs_; g++) {
      CUDA_CHECK(cudaFree(gpuTemp1_[g]));
      CUDA_CHECK(cudaFree(gpuTemp2_[g]));
    }

    delete[] gpuTemp1_;
    delete[] gpuTemp2_;

    gpuTemp1_ = nullptr;
    gpuTemp2_ = nullptr;
  } else {
    CUDA_CHECK(cudaFree(temp_));
    CUDA_CHECK(cudaFree(result_));
  
    temp_ = nullptr;
    result_ = nullptr;  
  }
}