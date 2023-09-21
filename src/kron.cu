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

#include "utils.h"
#include "kron.h"

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
  int kEqVar = 0; (maxColsAShape.ColsA == shape.ColsA) ? 1 : 0;
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
}

//Launch cuda kernels
template<typename T, uint NumFusedKerns>
cudaError_t generalSlicedMatmul(KernelInfo& kernelInfo, const uint kronIndex, T* x, T* kronMat[NumFusedKerns], 
                                T* kronGemmResult,
                                const uint M, const uint N, const uint K, 
                                const uint KronMatCols[NumFusedKerns], const uint KronMatRows[NumFusedKerns],
                                cudaStream_t stream) {
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
  typedef void (*KronMatmulKernel)(KernelParams<T, NumFusedKerns>);
  //Create kernel args;
  // void *args[] = {(void*)&params};
  ((KronMatmulKernel)kernelInfo.kernel)<<<grid, block,0,stream>>>(params);
  CUDA_CHECK(cudaDeviceSynchronize());
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
  for (uint i = 0; i < NumKronMats; i += MaxFusedKerns) {
    const uint kronMat = NumKronMats - i - 1;
    const uint NumFusedKerns = min(MaxFusedKerns, NumKronMats - i);
    // printf("243: NumFusedKerns %d kronMat \n", NumFusedKerns);
    uint FusedKronMatCols[NumFusedKerns];
    uint FusedKronMatRows[NumFusedKerns];
    for (int k = 0; k < NumFusedKerns; k++) {
      FusedKronMatCols[k] = KronMatCols[kronMat - k];
      FusedKronMatRows[k] = KronMatRows[kronMat - k];
    }

    auto selectedKernel = selectKernel(KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                       K, M, NumFusedKerns});
    tunedSeries.push_back({selectedKernel, kronMat - NumFusedKerns, kronMat, 0.0f});
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
  std::cout << "M " << M << " N " << N << " K " << K << std::endl;
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

    std::cout << "kernel selected " << selectedKernel << std::endl;
    std::cout << "prevTempN " << prevTempN << " currTempN " << currTempN << std::endl;
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
    std::cout << kronMat << std::endl;
    
    // if (kronMat >= 1)
    // printGPUArray<float>(M, currTempN, (kronMat == 2) ? 8.0f : (kronMat == 1 ? 64.0f : 512.0f),
    //                      (float*)currKronResult, stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    prevTempN = currTempN;
    std::cout << "prevTempN " << prevTempN << " currTempN " << currTempN << std::endl;
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

float minExecTimeOfSeries(uint M, uint N, uint K, const uint NumKronMats, 
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

    KronMatmulShape shape = KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                            K, M, NumFusedKerns};
    if (bestKernels.find(shape) == bestKernels.end()) continue;
    auto iter = bestKernels.find(shape);
    TunedKernelsSeries epilogueKernels;
    float kernelTime = iter->second.second;
    float epilogueTime = minExecTimeOfSeries(M, N, K, NumKronMats, KronMatCols, KronMatRows, 
                                             endKron + 1, epilogueKernels, bestKernels);
    if (minTime > kernelTime + epilogueTime) {
      minTime = kernelTime + epilogueTime;
      minEpilogueKernels = epilogueKernels;
      minPrologueKernel = TunedKernelFromStart(iter->second.first, startKron, endKron, kernelTime);
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
  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return cudaErrorInvalidValue;
  T* kronGemmResults[2] = {(T*)handle.temp_, (T*)handle.result_};
  T* prevKronResult = x;
  T* currKronResult = kronGemmResults[0];
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
    cudaError_t status;
    KronMatmulShape shape = KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                            K, M, NumFusedKerns};
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
          switch(NumFusedKerns) {
            case 1:
              status = generalSlicedMatmul<T, 1>(kernel, endKron, prevKronResult,
                                                krons, currKronResult, M, N, K, 
                                                FusedKronMatCols, FusedKronMatRows,
                                                stream);
              break;
            case 2:
              status = generalSlicedMatmul<T, 2>(kernel, endKron, prevKronResult,
                                                krons, currKronResult, M, N, K, 
                                                FusedKronMatCols, FusedKronMatRows,
                                                stream);
              break;
            case 3:
              status = generalSlicedMatmul<T, 3>(kernel, endKron, prevKronResult,
                                                krons, currKronResult, M, N, K, 
                                                FusedKronMatCols, FusedKronMatRows,
                                                stream);
              break;
            case 4:
              status = generalSlicedMatmul<T, 4>(kernel, endKron, prevKronResult,
                                                krons, currKronResult, M, N, K,
                                                FusedKronMatCols, FusedKronMatRows,
                                                stream);
              break;
            case 5:
              status = generalSlicedMatmul<T, 5>(kernel, endKron, prevKronResult,
                                                krons, currKronResult, M, N, K,
                                                FusedKronMatCols, FusedKronMatRows,
                                                stream);
              break;
            default:
                std::cout << "Invalid number of fused kernels" << std::endl;
              status = cudaErrorInvalidValue;
          }
          if (status != cudaSuccess) break;
        }
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        if (status != cudaSuccess)
          std::cout << "Error: " << cudaGetErrorString(status) << std::endl;
        float kernelTime;
        CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, end));
        std::cout << kernel << " runs in " << kernelTime << std::endl;
        if (kernelTime < minTime) {
          bestKernel = kernel;
          minTime = kernelTime;
        }
        if (status != cudaSuccess) return status;
      }
    }

    if (minTime < std::numeric_limits<float>::max()) {
      std::cout << "Best kernel for " << shape << ": " << bestKernel << "runs in " << (minTime/runs) << " ms" << std::endl;
      bestKernels.emplace(std::make_pair(shape, std::make_pair(bestKernel, minTime/runs)));
    }
  }}
  std::cout << "Finding min execution time of the series" << std::endl;
  TunedKernelsSeries tunedKernels;
  float minTime = minExecTimeOfSeries(M, N, K, NumKronMats,
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

template<typename T>
struct ThreadArgs {
  ThreadArgs() {}
  ThreadArgs(FastKronHandle* handle, uint NumKronMats, T* x, T** kronMats, T** result, 
            uint M, uint N, uint K, uint *KronMatCols, uint *KronMatRows, cudaStream_t* stream,
            uint gpuRow, uint gpuCol, uint gpusInRows, uint gpusInCols, pthread_barrier_t* barrier) : 
            handle(handle), NumKronMats(NumKronMats), x(x), kronMats(kronMats), result(result),
            M(M), N(N), K(K), KronMatCols(KronMatCols), KronMatRows(KronMatRows), stream(stream),
            gpuRow(gpuRow), gpuCol(gpuCol), gpusInRows(gpusInRows), gpusInCols(gpusInCols), barrier(barrier) {}

  FastKronHandle* handle;
  uint NumKronMats;
  T* x;
  T** kronMats;
  T** result;
  uint M;
  uint N;
  uint K;
  uint *KronMatCols;
  uint *KronMatRows;
  cudaStream_t* stream;
  uint gpuRow;
  uint gpuCol;
  uint gpusInRows;
  uint gpusInCols;
  pthread_barrier_t* barrier;

  struct ThreadResult {
    cudaError_t status;
    void* result;
  } threadResult;
};

template<typename T, typename VecT>
void singleGPUOutOfCoreThreadFunc(ThreadArgs<T>& thArgs) {
  // ThreadArgs<T>& thArgs = *(ThreadArgs<T>*)arg;

  FastKronHandle& handle = *thArgs.handle;
  uint NumKronMats = thArgs.NumKronMats;
  T* x = thArgs.x;
  T** kronMats = thArgs.kronMats;
  T** result = thArgs.result;
  uint M = thArgs.M;
  uint N = thArgs.N;
  uint K = thArgs.K;
  uint *KronMatCols = thArgs.KronMatCols;
  uint *KronMatRows = thArgs.KronMatRows;
  cudaStream_t* stream = thArgs.stream;
  uint gr = thArgs.gpuRow;
  uint gc = thArgs.gpuCol;
  uint gpusInRows = thArgs.gpusInRows;
  uint gpusInCols = thArgs.gpusInCols; 

  uint g = gr * gpusInCols + gc;
  CUDA_CHECK(cudaSetDevice(g));

  cudaError_t status;

  const uint uvaRows = handle.OutofCoreRows_;
  const uint uvaColsX = handle.OutofCoreKronBatch_ * power(KronMatRows[0], handle.OutofCoreKrons_); //KronMatCols[0] * KronMatCols[0]* KronMatCols[0]* KronMatCols[0] * KronMatCols[0] * KronMatCols[0];
  const uint batchedKronMuls = handle.OutofCoreKrons_;
  T* kronGemmResults[2] = {(T*)handle.temp_, (T*)handle.result_};
  T* outerPrevResult, *outerCurrResult;

  for (uint startOutofCoreRows = handle.OutofCoreRows_ * gr;
        startOutofCoreRows  < M; 
        startOutofCoreRows += handle.OutofCoreRows_ * gpusInRows) {
  const uint outOfCoreRows = min(handle.OutofCoreRows_, M - startOutofCoreRows);

  outerPrevResult = x;
  outerCurrResult = kronGemmResults[0];
  
  for (uint io = 0; io < NumKronMats; io += batchedKronMuls) {
    T* innerResults[2] = {(T*)handle.outOfCoreTemp1_[g], (T*)handle.outOfCoreTemp2_[g]};
    T* innerPrevResult = innerResults[0];
    T* innerCurrResult = innerResults[1];

    if (uvaColsX == K) {
      innerResults[0] = &kronGemmResults[0][startOutofCoreRows * K];
      innerResults[1] = &kronGemmResults[1][startOutofCoreRows * K];
      innerPrevResult = &x[startOutofCoreRows * K];
      innerCurrResult = innerResults[0];
    }

    uint KronMulBatchSize = min(batchedKronMuls, NumKronMats - io);
    uint MaxI = io + KronMulBatchSize;

    for (uint uvaPart = gc * uvaColsX; uvaPart < K; uvaPart += uvaColsX * gpusInCols) {
      //Copy outerPrevResult to innerPrevResult
      if (uvaColsX < K) {
        // CUDA_CHECK(cudaDeviceSynchronize());
        // printf("copyXtoUVAX\n");
        dim3 grid = {outOfCoreRows, 1,1};
        dim3 block = {256, 1, 1};
        copyXtoUVAX<T, VecT, 256><<<grid, block, 0, stream[g]>>>(M, N, K, KronMatRows[io], KronMatCols[io], innerPrevResult, outOfCoreRows, uvaColsX,
                                                                  &outerPrevResult[startOutofCoreRows * K], uvaPart/uvaColsX);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // printf("Done\n");
      }

      for (uint i = io; i < MaxI; i++) {
        const uint kronMat = NumKronMats - i - 1;
        // printf("g %d, kronMat %d %p\n", g, kronMat, kronMats[g * NumKronMats + kronMat]);
        printf("TODO:");
        // cudaError_t status = generalSlicedMatmul<T, VecT>(i, innerPrevResult, 
        //     kronMats[g * NumKronMats + kronMat], innerCurrResult, outOfCoreRows, uvaColsX, uvaColsX, 
        //     KronMatCols[kronMat], KronMatRows[kronMat], stream[g]);

        // printf("innerCurrResult %p innerPrevResult %p\n", innerCurrResult, innerPrevResult);
        if (status != cudaSuccess) goto end;

        //Double/ring/circular buffer previous result and new result
        if (i < MaxI - 1) {
          innerPrevResult = innerCurrResult;
          if (innerPrevResult == innerResults[0]) {        
            innerCurrResult = innerResults[1];
          } else if (innerPrevResult == innerResults[1]) {
            innerCurrResult = innerResults[0];
          }
        }
      }

      //Copy uvaCurrResult to kronGemmResult
      if (uvaColsX < K) {
        // CUDA_CHECK(cudaDeviceSynchronize());
        // printf("copyUVATempToY\n");
        dim3 grid = {outOfCoreRows, 1,1};
        dim3 block = {256, 1, 1};
        copyUVATempToY<T, VecT, 256><<<grid, block, 0, stream[g]>>>(M, N, K, KronMatRows[io], KronMatRows[io], innerCurrResult, outOfCoreRows, uvaColsX,
                                                                    &outerCurrResult[startOutofCoreRows * K], uvaPart/uvaColsX, KronMulBatchSize, io);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // printf("Done\n");
      } else {
        if (innerPrevResult == innerResults[0]) {
          outerPrevResult = kronGemmResults[0];
          outerCurrResult = kronGemmResults[1];
        }
        else {
          outerPrevResult = kronGemmResults[1];
          outerCurrResult = kronGemmResults[0];
        }

        // printf("outerPrevResult %p outerCurrResult %p\n", outerPrevResult, outerCurrResult);
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream[g]));
    int s = pthread_barrier_wait(thArgs.barrier);
    assert (s == 0 || s == PTHREAD_BARRIER_SERIAL_THREAD);

    //Double/ring/circular buffer previous result and new result
    if (io < NumKronMats - batchedKronMuls) {
      outerPrevResult = outerCurrResult;
      if (outerPrevResult == kronGemmResults[0]) {        
        outerCurrResult = kronGemmResults[1];
      } else if (outerPrevResult == kronGemmResults[1]) {
        outerCurrResult = kronGemmResults[0];
      }
    }
  }
  }

  end:
  thArgs.threadResult = {status, (void*)outerCurrResult};
}

template<typename T, typename VecT>
cudaError_t singleGPUOutOfCoreKronMatmul(FastKronHandle& handle, const uint NumKronMats, T* x, T* kronMats[], T** result,
                                         uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                                         cudaStream_t stream[]) {
  if (result == NULL)                       return cudaErrorInvalidValue;
  if (handle.OutofCoreRows_ > M)            return cudaErrorInvalidValue;
  if (NumKronMats < handle.OutofCoreKrons_) return cudaErrorInvalidValue;

  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return cudaErrorInvalidValue;
  
  const uint uvaRows = handle.OutofCoreRows_;
  const uint uvaColsX = handle.OutofCoreKronBatch_ * power(KronMatRows[0], handle.OutofCoreKrons_); //KronMatCols[0] * KronMatCols[0]* KronMatCols[0]* KronMatCols[0] * KronMatCols[0] * KronMatCols[0];
  const uint batchedKronMuls = handle.OutofCoreKrons_;

  // printf("MaxInnerKrons %d uvaColsX %d K %d handle.outOfCoreTemp1_ %p\n", handle.OutofCoreKrons_, uvaColsX, K, handle.outOfCoreTemp1_);
  double timeStart = getCurrTime();

  uint gpusInRows = 2;
  uint gpusInCols = 2; //handle.numGPUs_;
  
  assert(handle.numGPUs_ == gpusInRows * gpusInCols);
  assert(gpusInRows * handle.OutofCoreRows_ <= M);
  assert(M % (gpusInRows * handle.OutofCoreRows_) == 0);
  
  // pthread_t threads[handle.numGPUs_];
  //All gpus with same row shares the same barrier
  pthread_barrier_t barriers[gpusInRows];

  for (int i = 0; i < gpusInRows; i++) {
    int s = pthread_barrier_init(&barriers[i], NULL, gpusInRows);
    assert (s == 0);
  }

  std::thread* threads = new std::thread[handle.numGPUs_];

  ThreadArgs<T>* threadArgs = new ThreadArgs<T>[handle.numGPUs_];

  for (uint thread = 0; thread < handle.numGPUs_; thread++) {
    ThreadArgs<T> args = ThreadArgs<T>(
      &handle,
      NumKronMats,
      x,
      kronMats,
      result,
      M,
      N,
      K,
      &KronMatCols[0],
      &KronMatRows[0],
      stream,
      thread/gpusInCols,
      thread % gpusInCols,
      gpusInRows,
      gpusInCols,
      &barriers[thread/gpusInCols]
    );

    threadArgs[thread] = args;
    threads[thread] = std::thread(singleGPUOutOfCoreThreadFunc<T, VecT>, std::ref(threadArgs[thread]));
    
    // int s = pthread_create(&threads[thread], NULL, 
    //                    singleGPUOutOfCoreThreadFunc<T, VecT>,
    //                    (void *)&threadArgs[thread]);
    // assert (s == 0);
      // return cudaErrorInitializationError;
  }

  cudaError_t status;
  for (uint thread = 0; thread < handle.numGPUs_; thread++) {
    threads[thread].join();

    if (thread == 0) {
      status = threadArgs[thread].threadResult.status;
      *result = (T*)threadArgs[thread].threadResult.result;
    }
  }
  double timeEnd = getCurrTime();

  printf("531: time %lf microseconds\n", timeEnd - timeStart);
  // 
  // printf("*result %p\n", *result);
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

cudaError_t kronSGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]) {
  return singleGPUOutOfCoreKronMatmul<float, float4>(handle, NumKronMats, x, kronMats, result, 
                                                     M, N, K, KronMatCols, KronMatRows, stream);
}

cudaError_t kronIGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], int** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]) {
  return singleGPUOutOfCoreKronMatmul<int, int4>(handle, NumKronMats, x, kronMats, result, 
                                                 M, N, K, KronMatCols, KronMatRows, stream);
}

template<typename T> void FastKronHandle_init(FastKronHandle& handle, bool useUVA) {
  if (useUVA) {
    size_t sz = handle.M_ * handle.N_ * sizeof(T);
    CUDA_CHECK(cudaMallocManaged(&handle.temp_, sz));
    CUDA_CHECK(cudaMallocManaged(&handle.result_, sz));
    CUDA_CHECK(cudaMemset(handle.temp_, 0, sz));
    CUDA_CHECK(cudaMemset(handle.result_, 0, sz));
    
    assert(handle.numGPUs_ != 0);

    uint outOfCoreCols = handle.OutofCoreKronBatch_  * power(handle.KronMatCols_[0], handle.OutofCoreKrons_);
    if (outOfCoreCols < handle.K_) {
      size_t outOfCoreSz = handle.OutofCoreRows_ * outOfCoreCols * sizeof(T);

      handle.outOfCoreTemp1_ = new void*[handle.numGPUs_];
      handle.outOfCoreTemp2_ = new void*[handle.numGPUs_];

      for (int g = 0; g < handle.numGPUs_; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMalloc(&handle.outOfCoreTemp1_[g], outOfCoreSz));
        CUDA_CHECK(cudaMalloc(&handle.outOfCoreTemp2_[g], outOfCoreSz));
        CUDA_CHECK(cudaMemset(handle.outOfCoreTemp1_[g], 0, outOfCoreSz));
        CUDA_CHECK(cudaMemset(handle.outOfCoreTemp2_[g], 0, outOfCoreSz));
      }
    } else {
      handle.outOfCoreTemp1_ = handle.outOfCoreTemp2_ = nullptr;
    }
  } else {
    size_t tempN = handle.K_;
    size_t maxTempN = tempN;
    for (int i = 0; i < handle.NumKronMats_; i++) {
      tempN = (tempN/handle.KronMatRows_[i])*handle.KronMatCols_[i];
      if (maxTempN < tempN)
        maxTempN = tempN;
    }

    size_t sz = handle.M_ * maxTempN * sizeof(T);
    std::cout << "877: " << sz << std::endl;
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

template<> void FastKronHandle::init<float>(bool useUVA) {
  FastKronHandle_init<float>(*this, useUVA);
}

template<> void FastKronHandle::init<int>(bool useUVA) {
  FastKronHandle_init<int>(*this, useUVA);
}

template<> void FastKronHandle::init<double>(bool useUVA) {
  FastKronHandle_init<double>(*this, useUVA);
}

void FastKronHandle::free() {
  CUDA_CHECK(cudaFree(temp_));
  CUDA_CHECK(cudaFree(result_));

  temp_ = nullptr;
  result_ = nullptr;

  if (outOfCoreTemp1_ != nullptr) {
    for (uint g = 0; g < numGPUs_; g++) {
      CUDA_CHECK(cudaFree(outOfCoreTemp1_[g]));
      CUDA_CHECK(cudaFree(outOfCoreTemp2_[g]));
    }

    delete[] outOfCoreTemp1_;
    delete[] outOfCoreTemp2_;

    outOfCoreTemp1_ = nullptr;
    outOfCoreTemp2_ = nullptr;
  }
}