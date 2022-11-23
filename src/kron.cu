#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <type_traits>

#include "kron.h"

#define CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define DIVUP(x, y) (((x) + (y) - 1)/((y)))

#define EXTERNAL_KP_K_TILE_ 128

#define C_IN_REG
#define EVAL

//utils.h
static constexpr int log2(uint n) {return 31 - __builtin_clz(n);}
static constexpr int log2(int n) {return 31 - __builtin_clz(n);}

#include "kron_device.cu"

#define N_THREADS 256
#define KP_N_TILE 32

#define TILE_X 1

#define K_EQUALS_VAR_KERNELS(T, VecT, N_COARSE_TB, MAX_K, KP_N_K, K_EQUALS_VAR) \
  (void*)kronGemmKernel<T, VecT, N_THREADS,N_COARSE_TB,TILE_X,MAX_K,KP_N_K,KP_N_K,KP_N_TILE,K_EQUALS_VAR,1>,
  // (void*)cuda_gemm<DATA_TYPE,N_THREADS,N_COARSE_TB,TILE_X,MAX_K,KP_N_K,KP_N_K,KP_N_TILE,K_EQUALS_VAR,0>,

#define KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, KP_N_K) \
  K_EQUALS_VAR_KERNELS(T, VecT, N_COARSE_TB, MAX_K, KP_N_K, 0) \
  K_EQUALS_VAR_KERNELS(T, VecT, N_COARSE_TB, MAX_K, KP_N_K, 1)

#define MAX_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K) \
KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 64) \
  // KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 2) \
  // KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 4) \
  // KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 8) \
  // KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 16) \
  // KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 32) \
  // KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 64) \
//   KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 128) 

#define COARSE_TB_KERNELS(T, VecT, N_COARSE_TB) \
MAX_K_KERNELS(T, VecT, N_COARSE_TB, 4096) \
// MAX_K_KERNELS(T, VecT, N_COARSE_TB, 2048) \
// MAX_K_KERNELS(T, VecT, N_COARSE_TB, 4096) \

  // MAX_K_KERNELS(T, VecT, N_COARSE_TB, 16) \
  // MAX_K_KERNELS(T, VecT, N_COARSE_TB, 32) \
  // MAX_K_KERNELS(T, VecT, N_COARSE_TB, 64) \
  // MAX_K_KERNELS(T, VecT, N_COARSE_TB, 128) \
  // MAX_K_KERNELS(T, VecT, N_COARSE_TB, 256) \
  // MAX_K_KERNELS(T, VecT, N_COARSE_TB, 512) \
  MAX_K_KERNELS(T, VecT, N_COARSE_TB, 1024) \
  MAX_K_KERNELS(T, VecT, N_COARSE_TB, 2048) \
  MAX_K_KERNELS(T, VecT, N_COARSE_TB, 4096) \
  // MAX_K_KERNELS(T, VecT, N_COARSE_TB, 8192) \
  // MAX_K_KERNELS(T, VecT, N_COARSE_TB, 16384) 

#define TYPE_KERNELS(T, VecT) \
  COARSE_TB_KERNELS(T, VecT, 1)

//Two type kernels float/float4 and int/int4

#define NUM_TYPE_KERNELS 2
#define MIN_K 4096
#define MAX_K 4096
#define NUM_MAX_K_KERNELS (log2(MAX_K)-log2(MIN_K) + 1)

#define MIN_KP_K 64
#define MAX_KP_K 64
#define NUM_KP_N_K_KERNELS (log2(MAX_KP_K)-log2(MIN_KP_K) + 1)

#define NUM_COARSE_TB_KERNELS 1
#define NUM_K_EQUALS_VAR 2
#define NUM_KPK_EQUALS_VAR 1

static void* KronGemmKernels[NUM_TYPE_KERNELS][NUM_COARSE_TB_KERNELS][NUM_MAX_K_KERNELS][NUM_KP_N_K_KERNELS][NUM_K_EQUALS_VAR][NUM_KPK_EQUALS_VAR] = {
  // KP_N_K_KERNELS(8, 1024, 32)
  TYPE_KERNELS(float,  float4)
  TYPE_KERNELS(int,    int4)
  // TYPE_KERNELS(double, double4)
    // COARSE_TB_KERNELS(1)
    // COARSE_TB_KERNELS(2)
    // COARSE_TB_KERNELS(4)
  };

static_assert(sizeof(KronGemmKernels)/sizeof(void*) == NUM_TYPE_KERNELS * NUM_COARSE_TB_KERNELS * NUM_KP_N_K_KERNELS * NUM_MAX_K_KERNELS*NUM_K_EQUALS_VAR*NUM_KPK_EQUALS_VAR);

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

//Launch cuda kernels
template<typename T, typename VecT>
cudaError_t generalKronGemm(const uint NumKronMats, 
                            T* kronGemmResults[], T* x, T* kronMats[], T** kronGemmResult,
                            const uint M, const uint N, const uint K, 
                            const uint KronMatCols[], const uint KronMatRows[], 
                            cudaStream_t stream) {
  typedef int (*KronGemmKernel)(const uint, const uint, const uint, const uint, const uint, T*, T*, T*);
  cudaError_t status;

  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return cudaErrorInvalidValue;

  //Only row major layout of all matrics is supported.
  
  //Use double buffering for writing result and using output 
  //of previous iteration as input to current
  
  *kronGemmResult = kronGemmResults[0];
  T* prevResult = x;

  for (uint i = 0; i < NumKronMats; i++) {
    KronGemmKernel cuda_gemm_func = NULL;
    dim3 grid;
    dim3 block;

    const int KP_K_BATCH = 1;
    int N_COARSE_TB = 1; //(M > 100) ? 2 : 1;
    int min_k = min(K, MAX_K);
    int k_equals_var = (min_k == K) ? 1 : 0;
    // if (min_k/KronMatRows[0] >= 256) {
    //   //K dimension is very high. Divide it in different threadblocks to have better parallelism
    //   min_k = min_k/KronMatRows[0];
    //   k_equals_var = 0;
    // }
    // printf("min_k %d\n", min_k);
    uint typeKernelIdx = typeKernelIndex((T)0);
    
    //Check that kernel index is valid only in debug mode
    assert(typeKernelIdx < NUM_TYPE_KERNELS);
    assert(N_COARSE_TB/2 < NUM_COARSE_TB_KERNELS);
    assert(log2(min_k)-log2(MIN_K) < NUM_MAX_K_KERNELS);
    assert(log2(KronMatRows[0])-log2(MIN_KP_K) < NUM_KP_N_K_KERNELS);

    cuda_gemm_func = (KronGemmKernel)KronGemmKernels[typeKernelIdx][N_COARSE_TB/2][log2(min_k)-log2(MIN_K)][log2(KronMatRows[0])-log2(MIN_KP_K)][k_equals_var][0];
    
    assert(cuda_gemm_func != NULL);
    
    //Create the grid and thread block
    grid = {
              DIVUP((M/TILE_X), N_COARSE_TB),
              (K/min_k) * DIVUP(KronMatCols[0], KP_N_TILE), 
              DIVUP(KronMatRows[0], EXTERNAL_KP_K_TILE_)
           };
    block = {
              N_THREADS, 
              1, 
              1
            };
    
    //Create kernel args;
    void *args[] = {
                    (void*)&M, (void*)&N, (void*)&K, 
                    (void*)&KronMatRows[NumKronMats-i-1],
                    (void*)&KronMatCols[NumKronMats-i-1],
                    &prevResult, 
                    (void*)&kronMats[NumKronMats-i-1], 
                    (void*)kronGemmResult, 
                    &i
                  };

    status = cudaLaunchKernel((const void*)cuda_gemm_func, grid, block, &args[0], 0, stream);
    if (status != cudaSuccess)
      return status;

    //Double/ring/circular buffer previous result and new result
    if (i < NumKronMats - 1) {
      prevResult = *kronGemmResult;
      if (prevResult == kronGemmResults[0]) {        
        *kronGemmResult = kronGemmResults[1];
      } else if (prevResult == kronGemmResults[1]) {
        *kronGemmResult = kronGemmResults[0];
      }
    }
    
    // CUDA_CHECK(cudaDeviceSynchronize());
  }

  return status;
}

/**************************************************
          Library Functions
***************************************************/
cudaError_t kronSGEMM(const uint NumKronMats, float* kronGemmResults[], float* x, float* kronMats[], float** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream) {
  if (result == NULL) return cudaErrorInvalidValue;
  return generalKronGemm<float, float4>(NumKronMats, kronGemmResults, x, kronMats, result, M, N, K, KronMatCols, KronMatRows, stream);
}

cudaError_t kronIGEMM(const uint NumKronMats, int* kronGemmResults[], int* x, int* kronMats[], int** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream) {
  if (result == NULL) return cudaErrorInvalidValue;
  return generalKronGemm<int, int4>(NumKronMats, kronGemmResults, x, kronMats, result, M, N, K, KronMatCols, KronMatRows, stream);
}

cudaError_t kronDGEMM(const uint NumKronMats, double* kronGemmResults[], double* x, double* kronMats[], double** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream) {
  if (result == NULL) return cudaErrorInvalidValue;
  return generalKronGemm<double, double4>(NumKronMats, kronGemmResults, x, kronMats, result, M, N, K, KronMatCols, KronMatRows, stream);
}
