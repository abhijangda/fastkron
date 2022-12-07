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

#define C_IN_REG
#define EVAL

//utils.h
static constexpr int log2(uint n) {return 31 - __builtin_clz(n);}
static constexpr int log2(int n) {return 31 - __builtin_clz(n);}

#define N_THREADS 256

#include "kernel_decl.inc" 

#define TYPE_KERNELS(T, VecT) \
  KERNEL_DECL(T, VecT, 0),\
  KERNEL_DECL(T, VecT, 1),


//Three type kernels float/float4, int/int4, and double/double4
#define NUM_TYPE_KERNELS 3
// #define MIN_K 16
// #define MAX_K 4096
#define NUM_MAX_K_KERNELS (log2(MAX_K)-log2(MIN_K) + 1)

// #define MIN_KP_K 2
// #define MAX_KP_K 64
#define NUM_KP_N_K_KERNELS (log2(MAX_KP_K)-log2(MIN_KP_K) + 1)

#define NUM_COARSE_TB_KERNELS 1
#define NUM_K_EQUALS_VAR 2
#define NUM_KPK_EQUALS_VAR 1

#define EXTERNAL_KP_K_TILE_ MAX_K

#include "kron_device.cu"

static void* KronGemmKernels[NUM_TYPE_KERNELS][NUM_K_EQUALS_VAR][NUM_COARSE_TB_KERNELS][NUM_MAX_K_KERNELS][NUM_KP_N_K_KERNELS][NUM_KPK_EQUALS_VAR] = {
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
    const uint kronMat = NumKronMats-i-1;

    const int KP_K_BATCH = 1;
    int N_COARSE_TB = 1; //(M > 100) ? 2 : 1;
    int max_k;
    int min_k;
    int max_k_kernel = 1;
    
    // if (min_k/KronMatRows[0] >= 256) {
    //   //K dimension is very high. Divide it in different threadblocks to have better parallelism
    //   min_k = min_k/KronMatRows[0];
    //   k_equals_var = 0;
    // }
    // printf("min_k %d\n", min_k);
    uint typeKernelIdx = typeKernelIndex((T)0);
    if (KronMatCols[kronMat] == 1024) {
      min_k = MAX_K;
    } else if (KronMatCols[kronMat] >= 256) {
      min_k = 32768;
    } else {
      while (max_k_kernel < MIN_K) {
        max_k_kernel *= KronMatCols[0];
      }
      while (max_k_kernel < MAX_K && KronGemmKernels[typeKernelIdx][0][0][log2(max_k_kernel)-log2(MIN_K)][log2(KronMatRows[0])-log2(MIN_KP_K)][0] != NULL) {
        // printf("max_k_kernel %d KronMatCols[0] %d\n", max_k_kernel, KronMatCols[0]);
        max_k_kernel *= KronMatCols[0];
      }

      // printf("max_k_kernel %d\n", max_k_kernel);

      if (max_k_kernel > MAX_K || KronGemmKernels[typeKernelIdx][0][0][log2(max_k_kernel)-log2(MIN_K)][log2(KronMatRows[0])-log2(MIN_KP_K)][0] == NULL)
        max_k_kernel = max_k_kernel/KronMatCols[0];

      // printf("max_k_kernel %d\n", max_k_kernel);


      if (K > max_k_kernel) {
        max_k = 1;
        while (max_k <= max_k_kernel)
          max_k *= KronMatCols[kronMat];
        
        max_k = max_k/KronMatCols[kronMat];
        min_k = min(K, max_k);
      } else {
        min_k = K;
      }
    }

    int k_equals_var = (min_k == K) ? 1 : 0;
    // printf("min_k %d k_equals_var %d\n", min_k, k_equals_var);

    //Check that kernel index is valid only in debug mode
    assert(typeKernelIdx < NUM_TYPE_KERNELS);
    assert(N_COARSE_TB/2 < NUM_COARSE_TB_KERNELS);
    assert(log2(min_k)-log2(MIN_K) < NUM_MAX_K_KERNELS);
    assert(log2(KronMatRows[0])-log2(MIN_KP_K) < NUM_KP_N_K_KERNELS);

    cuda_gemm_func = (KronGemmKernel)KronGemmKernels[typeKernelIdx][k_equals_var][0][log2(min_k)-log2(MIN_K)][log2(KronMatRows[0])-log2(MIN_KP_K)][0];
    
    assert(cuda_gemm_func != NULL);
    uint tileRowA = MaxTileRowsA[log2(KronMatRows[kronMat])-log2(MIN_KP_K)];
    uint tileKronCols = MaxTileKronCols[log2(KronMatRows[kronMat])-log2(MIN_KP_K)];
    //Create the grid and thread block
    grid = {
              DIVUP(M, tileRowA),
              (K/min_k) * DIVUP(KronMatCols[kronMat], tileKronCols),
              1// DIVUP(KronMatRows[kronMat], EXTERNAL_KP_K_TILE_)
           };
    block = {
              N_THREADS, 
              1, 
              1
            };
    
    //Create kernel args;
    void *args[] = {
                    (void*)&M, (void*)&N, (void*)&K, 
                    (void*)&KronMatRows[kronMat],
                    (void*)&KronMatCols[kronMat],
                    &prevResult, 
                    (void*)&kronMats[kronMat], 
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
