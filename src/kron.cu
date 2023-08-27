#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <type_traits>
#include <thread>

#include <unordered_map>
#include <vector>
#include <iostream>

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

enum ElementType {
  Float,
  Double,
  Int,
  Long
};

struct KernelInfo {
  void* kernel;
  uint NumThreads;
  uint KronCols;
  uint KronRows;
  uint TileKronCols;
  uint TileRowsA;
  uint MaxColsA;
  uint CRegRows;
  uint CRegCols;
  uint NumFusedKerns;
  ElementType elemType;
  bool RowModTileIsZero;
  bool KEqVar;
};

enum RowParallelismTy {
  Low = 0,
  Medium,
  High,
  Num = 3,
};

struct KronMatmulShape {
  uint KronCols;
  uint KronRows;
  uint ColsA;

  bool operator==(const KronMatmulShape& other) const {
    return KronCols == other.KronCols && KronRows == other.KronRows &&
    ColsA == other.ColsA;
  }
};

template<>
struct std::hash<KronMatmulShape> {
  std::size_t operator()(const KronMatmulShape& k) const {
    return hash<uint>()(k.KronCols) ^ hash<uint>()(k.KronRows) ^ hash<uint>()(k.ColsA);
  }
};

//Map from Factor size and Number of factors to KernelInfos
static std::unordered_map<KronMatmulShape, std::vector<KernelInfo>> compiledKernels;

#define N_THREADS 256
// #define MaxFusedKerns 4

#include "kernel_decl.inc" 

#define TYPE_KERNELS(T, VecT, ElemType) \
  KERNEL_DECL(T, VecT, ElemType, 0, 0),\
  KERNEL_DECL(T, VecT, ElemType, 1, 0),\
  KERNEL_DECL(T, VecT, ElemType, 0, 1),\
  KERNEL_DECL(T, VecT, ElemType, 1, 1),

//Three type kernels float/float4, int/int4, and double/double4
#define NUM_TYPE_KERNELS 2
#define NUM_MAX_K_KERNELS (log2(MAX_K)-log2(MIN_K) + 1)
#define NUM_KP_N_K_KERNELS (log2(MAX_KP_K)-log2(MIN_KP_K) + 1)

#define NUM_K_EQUALS_VAR 2
#define NUM_KPK_EQUALS_VAR 1
#define NUM_ROWS_MOD_TILE_IS_ZERO 2
#define EXTERNAL_KP_K_TILE_ MAX_K

#include "kron_device.cu"

static KernelInfo KronGemmKernels[] = {
  // KP_N_K_KERNELS(8, 1024, 32)
  TYPE_KERNELS(float,  float4, ElementType::Float)
  // TYPE_KERNELS(int,    int4)
  // TYPE_KERNELS(double, double4)
    // COARSE_TB_KERNELS(1)
    // COARSE_TB_KERNELS(2)
    // COARSE_TB_KERNELS(4)
  };

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

uint maxCompiledColsA(KronMatmulShape shape, uint NumFusedKerns = -1) {
  while (compiledKernels.find(shape) == compiledKernels.end()) {
    shape.ColsA /= 2;
    if (shape.ColsA == 1) {
     break;
    }
  }
  if (NumFusedKerns != -1 && shape.ColsA > 1) {
    while (compiledKernels.find(shape) != compiledKernels.end()) {
      bool found = false;
      for (auto info : compiledKernels.find(shape)->second) {
        if (info.NumFusedKerns == NumFusedKerns) {
          found = true;
          break;
        }
      }
      if (found) break;
      shape.ColsA /= 2;
    }
  }

  if (shape.ColsA == 1)
    std::cout << "Error: Cannot find compiled kernel\n";
  return shape.ColsA;
}

uint maxFusedKernels(KronMatmulShape shape) {
  uint max_k = maxCompiledColsA(shape);
  auto iter = compiledKernels.find(KronMatmulShape{shape.KronCols, shape.KronRows, max_k});
  uint numFusedKernels = 1;
  for (auto info : iter->second) {
    numFusedKernels = std::max(numFusedKernels, info.NumFusedKerns);
  }
  return numFusedKernels;
}

//Launch cuda kernels
template<typename T, typename VecT, uint NumFusedKerns>
cudaError_t generalSlicedMatmul(const uint kronIndex, T* x, T* kronMat[NumFusedKerns], T* kronGemmResult,
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
  //Go through all MaxColsA starting from MAX_K and select the relevant
  min_k = maxCompiledColsA(KronMatmulShape{KronMatCols[0], KronMatRows[0], K}, NumFusedKerns);  
  int kEqVar = (min_k == K) ? 1 : 0;
  auto iter = compiledKernels.find(KronMatmulShape{KronMatCols[0], KronMatRows[0], min_k});
  if (iter == compiledKernels.end()) {
    std::cout << "No kernel found" << std::endl;
    return cudaErrorInvalidValue;
  }
  auto kernelInfos = iter->second;
  KernelInfo kernelInfo;
  for (auto info : kernelInfos) {
    //TODO: need to check for tupe
    if (info.KEqVar == kEqVar && info.NumFusedKerns == NumFusedKerns) {
      uint tileRowA = info.TileRowsA;
      bool row_mod_tile_zero = (M % tileRowA) == 0;    
      if (info.RowModTileIsZero == row_mod_tile_zero) {
        kernelInfo = info;
        break;
      }
    }
  }

  const uint NumThreads = kernelInfo.NumThreads;
  {
    const uint CRegRows = kernelInfo.CRegRows;
    const uint CRegCols = kernelInfo.CRegCols;
    const uint MaxColsA = kernelInfo.MaxColsA;
    const uint KronRows = kernelInfo.KronRows;
    uint c1 = MAX(1, NumThreads/((kernelInfo.MaxColsA/kernelInfo.KronRows)/CRegRows));
    
    if (kernelInfo.TileKronCols != c1 * CRegCols) {
      printf("Invalid configuration: TileKronCols %d != c1*CRegCols %d; NumThreads %d CRegRows %d CRegCols %d MaxColsA %d\n", 
              kernelInfo.TileKronCols, c1 * CRegCols, NumThreads, CRegRows, CRegCols, MaxColsA);
      abort();
    }
    if (MaxColsA/KronRows > kernelInfo.NumThreads*c1* kernelInfo.CRegRows) {
      printf("MaxColsA/KronRows %d kernelInfo.NumThreads*c1* kernelInfo.CRegRows %d\n", MaxColsA/KronRows, kernelInfo.NumThreads*c1* kernelInfo.CRegRows);
      printf("Invalid configuration: MaxColsA %d KronRows %d NumThreads %d CRegRows %d CRegCols %d\n",
              MaxColsA, KronRows, NumThreads, CRegRows, CRegCols);
      abort();
    }
  }

  //Create the grid and thread block
  grid = {
            (K/min_k) * DIVUP(KronMatCols[0], kernelInfo.TileKronCols),
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
  status = cudaGetLastError();
  // status = cudaLaunchKernel((const void*)cuda_gemm_func, grid, block, &args[0], 0, stream);
  // if (status != cudaSuccess)
  //   return status;
  return status;
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
  const uint MaxFusedKerns = handle.getUseFusion() ? maxFusedKernels(KronMatmulShape{KronMatCols[0], KronMatRows[0], K}) : 1;
  //Use double buffering for writing result and using output 
  //of previous iteration as input to current
  for (uint i = 0; i < NumKronMats; i += MaxFusedKerns) {
    const uint kronMat = NumKronMats - i - 1;
    const uint NumFusedKerns = min(MaxFusedKerns, NumKronMats - i);
    T* krons[NumFusedKerns];
    uint FusedKronMatCols[NumFusedKerns];
    uint FusedKronMatRows[NumFusedKerns];
    for (int k = 0; k < NumFusedKerns; k++) {
      krons[k] = kronMats[kronMat - k];
      FusedKronMatCols[k] = KronMatCols[kronMat - k];
      FusedKronMatRows[k] = KronMatRows[kronMat - k];
    }
    cudaError_t status;
    switch(NumFusedKerns) {
      case 1:
        status = generalSlicedMatmul<T, VecT, 1>(i, prevKronResult,
                                                 krons, currKronResult, M, N, K, 
                                                 FusedKronMatCols, FusedKronMatRows,
                                                 stream);
        break;
      case 2:
        status = generalSlicedMatmul<T, VecT, 2>(i, prevKronResult,
                                                 krons, currKronResult, M, N, K, 
                                                 FusedKronMatCols, FusedKronMatRows,
                                                 stream);
        break;
      case 3:
        status = generalSlicedMatmul<T, VecT, 3>(i, prevKronResult,
                                                 krons, currKronResult, M, N, K, 
                                                 FusedKronMatCols, FusedKronMatRows,
                                                 stream);
        break;
      case 4:
        status = generalSlicedMatmul<T, VecT, 4>(i, prevKronResult,
                                                 krons, currKronResult, M, N, K,
                                                 FusedKronMatCols, FusedKronMatRows,
                                                 stream);
        break;
      case 5:
        status = generalSlicedMatmul<T, VecT, 5>(i, prevKronResult,
                                                 krons, currKronResult, M, N, K,
                                                 FusedKronMatCols, FusedKronMatRows,
                                                 stream);
        break;
      default:
          std::cout << "Invalid number of fused kernels" << std::endl;
        status = cudaErrorInvalidValue;
    }

    if (status != cudaSuccess) return status;

    //Double/ring/circular buffer previous result and new result
    if (i < NumKronMats - MaxFusedKerns) {
      prevKronResult = currKronResult;
      if (prevKronResult == kronGemmResults[0]) {        
        currKronResult = kronGemmResults[1];
      } else if (prevKronResult == kronGemmResults[1]) {
        currKronResult = kronGemmResults[0];
      }
    }
  }

  *result = currKronResult;

  return cudaSuccess;
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
    size_t sz = handle.M_ * handle.N_ * sizeof(T);
    CUDA_CHECK(cudaMalloc(&handle.temp_, sz));
    CUDA_CHECK(cudaMalloc(&handle.result_, sz));
    CUDA_CHECK(cudaMemset(handle.temp_, 0, sz));
  }

  //Initialize compiledKernels map

  for (uint i = 0; i < sizeof(KronGemmKernels)/sizeof(KernelInfo); i++) {
    KernelInfo& info = KronGemmKernels[i];
    KronMatmulShape shape {info.KronCols, info.KronRows, info.MaxColsA};
    auto iter = compiledKernels.find(shape);
    if (iter == compiledKernels.end()) {
      compiledKernels.emplace(std::make_pair(shape, std::vector<KernelInfo>()));
    }
    compiledKernels.at(shape).push_back(info);
  }
  //TODO: Add if debug
  // std::cout << "KronMatmulShapes: " << compiledKernels.size() << std::endl;
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