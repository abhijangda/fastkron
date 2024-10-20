#include <iostream>
#include <string>

#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include "fastkron.h"
#ifdef ENABLE_MULTI_GPU
#include "fastkronMg.h"
#endif
#include "handle/handle.h"

#ifndef __TEST_BASE_H__
#define __TEST_BASE_H__

#ifdef TEST_BACKEND_CUDA                         
  #include <cuda.h>
  #include <cuda_runtime.h>

#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                    \
    if( e != cudaSuccess ) {                          \
      printf("Failed: Cuda error %s:%d '%s'\n",             \
          __FILE__,__LINE__,cudaGetErrorString(e));   \
      abort();                             \
    }                                                 \
  } while(0)

#else
#define CUDACHECK(cmd) {}
#endif

#ifdef TEST_BACKEND_HIP
  #include <hip/hip_common.h>
  #include <hip/hip_runtime.h>

  #define HIPCHECK(cmd) do {                         \
    hipError_t e = cmd;                    \
    if( e != hipSuccess ) {                          \
      printf("39\n");\
      printf("Failed: HIP error %s:%d '%s'\n",             \
          __FILE__,__LINE__,hipGetErrorString(e));   \
      abort();                             \
    }                                                 \
  } while(0)
#else
#define HIPCHECK(cmd) {}
#endif

// static double convertTimeValToDouble(struct timeval _time) {
//   return ((double)_time.tv_sec)*1e6 + ((double)_time.tv_usec);
// }

// static struct timeval getTimeOfDay () {
//   struct timeval _time;

//   if (gettimeofday (&_time, NULL) == -1) {
//     fprintf (stderr, "gettimeofday returned -1\n");
//     perror ("");
//     abort ();
//   }

//   return _time;
// }

// static double getCurrTime() {
//   return convertTimeValToDouble(getTimeOfDay());
// }

/**************************************************
                Matrix Functions
***************************************************/
int one(int, int) {return 1;}
int zero(int, int) {return 0;}
int zeroOne(int i, int) {return i % 2;}
int zeroOneJ(int, int j) {return j % 2;}
int setToI(int i, int) {return i;}
int setToJ(int, int j) {return j;}
int iPlusJ(int i, int j) {return i + j;}
int randMod(int, int) {return rand()%3 + 1;}

template<typename T>
static void setMatrix(T* mat, uint M, uint N, int (*fnvalue)(int i, int j)) {
  for (uint i = 0; i < M; i++) {    
    for (uint j = 0; j < N; j++) {
      int v = fnvalue(i,j);
      mat[i*N + j] = T(v);
    }
  }
}

template<typename T>
void setValues(uint NUM_KP_MATS, T* kpMats[], T *x, T* y, uint M, uint N, uint K, uint KP_MAT_N[], uint KP_MAT_K[], int batchCountX, int batchCountF, int batchCountY, int (*xval)(int i, int j), int (*fval)(int i, int j))
{
  for (uint i = 0; i < NUM_KP_MATS; i++) {
    for (int b = 0; b < batchCountF; b++) {
      setMatrix(&kpMats[i][(b * KP_MAT_K[i] * KP_MAT_N[i])],
                KP_MAT_K[i], KP_MAT_N[i], fval);
    }
  }

  for (int b = 0; b < batchCountX; b++)
    setMatrix(&x[b * M * K], M, K, xval);

  for (int b = 0; b < batchCountY; b++)
    setMatrix(&y[b*M*N], M, N, xval);
}

// static void printMatrix(int* mat, int M, int N, int max_rows = -1, int max_cols = -1) {
//   printf("[");
//   for (uint i = 0; i < M; i++) {
//     for (uint j = 0; j < N; j++) {
//       // if (mat[i*N + j] == 18496)
//         // printf("%d,%d\n",i,j);
//       if (max_cols != -1 && j >= max_cols)
//         break;  
//       printf("%d, ", mat[i*N + j]);
//     }
//     if (i < M-1)
//       printf("\n");
//     if (max_rows != -1 && i >= max_rows)
//       break;
//   }
//   printf("]");
// }

/**************************************************
          Equality Check Functions
***************************************************/
template<typename T> static inline bool eqVal(T x, T y) {abort(); printf("invalid type\n"); return false;}

template<> inline bool eqVal(int x, int y) {return x == y;}

template<> inline bool eqVal(float x, float y) {
  if (abs(x) <= 1e-5 && abs(y) <= 1e-5) return true;
  if (abs(y) <= 1e-5) return abs((x-y)/x) <= 1e-5;
  return abs((x-y)/y) <= 1e-5;
}

template<> inline bool eqVal(double x, double y) {
  if (abs(x) <= 1e-5 && abs(y) <= 1e-5) return true;
  if (abs(y) <= 1e-5) return abs((x-y)/x) <= 1e-5;
  return abs((x-y)/y) <= 1e-5;
}

template<typename T>
static inline bool check(T* ref, T* computed, uint batchCount, uint M, uint N) {
  for (uint b = 0; b < batchCount; b++) {
    for (uint i = 0; i < M; i++) {
      for (uint j = 0; j < N; j++) {
        if (!eqVal(ref[b * M*N+i*N + j], computed[b*M*N + i* N + j])) {
          std::cout << "Mismatch for " << M << " x " << N << " at (" << b << ", " << i << ", " << j << "): ref = " << ref[b*M*N+i*N+j] << " computed = " << computed[b*M*N+i*N+j] << "\n";
          return false;
        }
      }
    }
  }

  return true;
}

fastKronBackend getTestBackend() {
#ifdef TEST_BACKEND_CUDA
  return fastKronBackend_CUDA;
#elif defined(TEST_BACKEND_HIP)
  return fastKronBackend_HIP;
#elif defined(TEST_BACKEND_X86)
  return fastKronBackend_X86;
#elif defined(TEST_BACKED_ARM)
  return fastKronBackend_ARM;
#endif
}

#ifdef TEST_BACKEND_CUDA
  #define TEST_BACKEND CUDA
#elif defined(TEST_BACKEND_HIP)
  #define TEST_BACKEND HIP
#elif defined(TEST_BACKEND_X86)
  #define TEST_BACKEND X86
#elif defined(TEST_BACKEND_ARM)
  #define TEST_BACKEND ARM
#endif


#define EXPAND(X, Y) EXPAND_(X, Y)
#define EXPAND_(X, Y) X##Y

/**************************************************
              Serial KronGEMM Functions
***************************************************/

//Perform Kronecker multiplications to get full matrix and multiply
//that with other matrix
void baselineKPThenMatmul(uint NUM_KP_MATS, int* result, int* x, int* kpout[], int* kpMats[],
                          uint M, uint N, uint K, uint KP_MAT_N[], uint KP_MAT_K[]) {
  uint cols;
  uint rows;

  for (uint kp = 0; kp < NUM_KP_MATS - 1; kp++) {
    int* kpFirst = (kp == 0) ? kpMats[0] : kpout[kp - 1];
    uint kpFirstRows = (kp == 0) ? KP_MAT_K[0] : rows;
    uint kpFirstCols = (kp == 0) ? KP_MAT_N[0] : cols;

    cols = kpFirstCols * KP_MAT_N[kp+1];
    rows = kpFirstRows * KP_MAT_K[kp+1];
    for (uint i = 0; i < rows; i++) {
      for (uint j = 0; j < cols; j++) {
        int v2 = kpMats[kp+1][(i%KP_MAT_K[kp+1]) * KP_MAT_N[kp+1] + j%KP_MAT_N[kp+1]];
        int v1 = kpFirst[(i/KP_MAT_K[kp+1]) * kpFirstCols + j/KP_MAT_N[kp+1]];
        kpout[kp][i*cols + j] = v1 * v2;
      }
    }
  }

  for(uint i = 0; i < M; i++) {    
    for(uint j = 0; j < N; j++) {    
      result[i* N + j] = 0;    
      for(uint k = 0; k < K; k++) {   
        result[i * N + j] += x[i*K + k]*kpout[NUM_KP_MATS-2][k*N + j];
      }    
    }    
  }

  // printMatrix(result, M, N, 4, 4);
}

//Serial implementation of the new Kron GEMM implementation
template<typename T>
void slicedMatmul(FastKronMMType kmmtype, uint NUM_KP_MATS, T* kpMatmulResult[], T* x, T* kpMats[], T* y,
                  uint M, uint /*N*/, uint K, uint KP_MAT_N[], uint KP_MAT_K[],
                  uint64_t strideX, uint64_t strideZ, uint64_t strideF[], uint64_t strideY, int batchCount,
                  fastKronOp opx, fastKronOp opfs, T alpha, T beta) {
  fastKronOp opy = fastKronOp_N;
  if (kmmtype == FastKronMMType::MKM) {
  for (int b = 0; b < batchCount; b++) {
  uint secFacRowMulSize = 1;
  uint rowsTillNow = 1;
  uint colsTillNow = 1;
  uint resultCols = 0;
  for (uint kp = 0; kp < NUM_KP_MATS; kp++) {
    T* prevKPMatmul = (kp == 0) ? x : kpMatmulResult[kp - 1];
    uint kpSecondK = KP_MAT_K[NUM_KP_MATS - 1 - kp];
    uint kpSecondN = KP_MAT_N[NUM_KP_MATS - 1 - kp];
    int prevKPMatmulCols = (kp == 0) ? K : resultCols;
    
    resultCols = (prevKPMatmulCols/kpSecondK) * kpSecondN;
    secFacRowMulSize = (kp == 0) ? K/kpSecondK : rowsTillNow * (K/(colsTillNow * KP_MAT_K[NUM_KP_MATS - 1 - (kp)]));
    //Number of times a column is multiplied with input matrix is equal to 
    //N/(number of column elements of this matrix * cols so far) * number of rows so far.
    rowsTillNow *= KP_MAT_N[NUM_KP_MATS - 1 - (kp)];
    colsTillNow *= KP_MAT_K[NUM_KP_MATS - 1 - (kp)];

    #pragma omp parallel for collapse(2)
    for (uint i = 0; i < M; i++) {
      for (uint j = 0; j < resultCols; j++) {
        T r = 0;

        for (uint kp_k = 0; kp_k < kpSecondK; kp_k++) {
          uint slice = (j / secFacRowMulSize) % kpSecondN;

          T v2 = 0;
          if (opfs == fastKronOp_T) {
            v2 = kpMats[NUM_KP_MATS - 1 - kp][b*strideF[NUM_KP_MATS - 1 - kp] + slice*KP_MAT_K[NUM_KP_MATS - 1 - kp] + kp_k];
          } else {
            v2 = kpMats[NUM_KP_MATS - 1 - kp][b*strideF[NUM_KP_MATS - 1 - kp] + kp_k*kpSecondN + slice];
          }

          T v1;
          uint32_t stridePrevKPMatmul = (kp == 0) ? strideX : strideZ;
          if (opx == fastKronOp_T && kp == 0)
            v1 = prevKPMatmul[b * stridePrevKPMatmul + ((j*kpSecondK)%prevKPMatmulCols + kp_k) * M + i];
          else
            v1 = prevKPMatmul[b * stridePrevKPMatmul + i* prevKPMatmulCols + (j*kpSecondK)%prevKPMatmulCols + kp_k];
          r += v1 * v2;
        }
        if (kp < NUM_KP_MATS - 1)
          kpMatmulResult[kp][b*strideZ + i*resultCols + j] = r;
        else {
          kpMatmulResult[kp][b*strideZ + i*resultCols + j] = alpha * r + beta*y[b*strideY + i*resultCols + j];
        }
      }
    }
  }}}

  if (kmmtype == FastKronMMType::KMM) {
  for (int b = 0; b < batchCount; b++) {
  uint secFacRowMulSize = 1;
  uint rowsTillNow = 1;
  uint colsTillNow = 1;
  uint resultCols = 0;
  for (uint kp = 0; kp < NUM_KP_MATS; kp++) {
    T* prevKPMatmul = (kp == 0) ? x : kpMatmulResult[kp - 1];
    uint kpSecondK = KP_MAT_K[kp];
    uint kpSecondN = KP_MAT_N[kp];
    int prevKPMatmulCols = (kp == 0) ? K : resultCols;
    
    resultCols = (prevKPMatmulCols/kpSecondK) * kpSecondN;
    secFacRowMulSize = (kp == 0) ? K/kpSecondK : rowsTillNow * (K/(colsTillNow * KP_MAT_K[kp]));
    //Number of times a column is multiplied with input matrix is equal to 
    //N/(number of column elements of this matrix * cols so far) * number of rows so far.
    rowsTillNow *= KP_MAT_N[kp];
    colsTillNow *= KP_MAT_K[kp];

    #pragma omp parallel for collapse(2)
    for (uint j = 0; j < resultCols; j++) {
      for (uint i = 0; i < M; i++) {
        T r = 0;

        for (uint kp_k = 0; kp_k < kpSecondK; kp_k++) {
          uint slice = (j / secFacRowMulSize) % kpSecondN;

          T v2 = 0;
          if (opfs == fastKronOp_T) { printf("321 todo\n"); abort();
            // v2 = kpMats[NUM_KP_MATS - 1 - kp][b*strideF[NUM_KP_MATS - 1 - kp] + slice*KP_MAT_K[NUM_KP_MATS - 1 - kp] + kp_k];
          } else {
            v2 = kpMats[kp][b*strideF[kp] + kp_k + slice*kpSecondK];
          }

          T v1;
          uint32_t stridePrevKPMatmul = (kp == 0) ? strideX : strideZ;
          if (opx == fastKronOp_T && kp == 0) {printf("321 todo\n"); abort();
            v1 = prevKPMatmul[b * stridePrevKPMatmul + ((j*kpSecondK)%prevKPMatmulCols + kp_k) * M + i];}
          else
            v1 = prevKPMatmul[b * stridePrevKPMatmul + i + ((j*kpSecondK)%prevKPMatmulCols + kp_k) * M];
          r += v1 * v2;
        }
        if (kp < NUM_KP_MATS - 1)
          kpMatmulResult[kp][b*strideZ + i + j * M] = r;
        else {
          kpMatmulResult[kp][b*strideZ + i + j * M] = alpha * r + beta*y[b*strideY + i + j * M];
        }
      }
    }
  }}
  }
}

/**************************************************
              Call KronGEMM Library Functions
***************************************************/
template<typename T>
static void kronGEMM(fastKronHandle handle, const fastKronBackend backend, FastKronMMType kronmatmulType, const uint NUM_KP_MATS, T* x, fastKronOp opx, T* kpMats[], fastKronOp opfs, T* z, T* y, T alpha, T beta,
                     uint M, uint/*N*/, uint/*K*/, uint KP_MAT_N[], uint KP_MAT_K[], 
                     uint32_t batchCount, uint64_t strideX, uint64_t strideZ, uint64_t strideF[],
                     uint64_t strideY, T* temp1, T* temp2) {

  if (batchCount > 1) {
    if (std::is_same<T, float>::value) {
      FastKronCHECK(sgekmmStridedBatched(handle, backend, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N,  
                      (const float*)x, opx, strideX, (const float**)kpMats, opfs, strideF, (float*)y,
                      strideY, alpha, beta, batchCount, (const float*)z, strideZ, (float*)temp1, (float*)temp2));
    } else if (std::is_same<T, double>::value) {
      FastKronCHECK(dgekmmStridedBatched(handle, backend, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N,  
                      (const double*)x, opx, strideX, (const double**)kpMats, opfs, strideF, (double*)y,
                      strideY, alpha, beta, batchCount, (const double*)z, strideZ, (double*)temp1, (double*)temp2));
    }

    return;
  }

  if (kronmatmulType == FastKronMMType::MKM) {
    if (std::is_same<T, float>::value) {
      //TODO: Change KMM to MKM
      FastKronCHECK(sgemkm(handle, backend, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N,  
                      (const float*)x, opx, (const float**)kpMats, opfs, (float*)y,
                      alpha, beta, (const float*)z, (float*)temp1, (float*)temp2));
    } else if (std::is_same<T, int>::value) {
      FastKronCHECK(igemkm(handle, backend, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N,  
                      (const int*)x, opx, (const int**)kpMats, opfs, (int*)y,
                      alpha, beta, (const int*)z, (int*)temp1, (int*)temp2));
    } else if (std::is_same<T, double>::value) {
      FastKronCHECK(dgemkm(handle, backend, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N,  
                      (const double*)x, opx, (const double**)kpMats, opfs, (double*)y,
                      alpha, beta, (const double*)z, (double*)temp1, (double*)temp2));
    } else {
      printf("Invalid type\n");
      return;
    }
  } else if (kronmatmulType == FastKronMMType::KMM) {
    if (std::is_same<T, float>::value) {
      FastKronCHECK(sgekmm(handle, backend, NUM_KP_MATS, KP_MAT_N, KP_MAT_K, M, 
                      (const float**)kpMats, opfs, (const float*)x, opx, (float*)y,
                      alpha, beta, (const float*)z, (float*)temp1, (float*)temp2));
    }
  }

  return;
}

#ifdef TEST_BACKEND_CUDA
template<typename T>
static void kronDistributedGEMM(fastKronHandle handle, const uint NUM_KP_MATS, T* x[], T* kpMats[], T* result[],
            uint M, uint N, uint K, uint KP_MAT_N[], uint KP_MAT_K[], 
            T* temp1[], T* temp2[], cudaStream_t stream[]) {
#ifdef ENABLE_MULTI_GPU
  if (std::is_same<T, float>::value) {
    FastKronCHECK(fastKronMgSGEMM(handle, NUM_KP_MATS,
                                  (void**)x, (void**)kpMats, (void**)result,
                                  M, N, K, KP_MAT_N, KP_MAT_K, 
                                  (void**)temp1, (void**)temp2, 
                                  (void*)stream));
  } else if (std::is_same<T, int>::value) {
    // CUDACHECK(kronDistributedSGEMM(handle, NUM_KP_MATS,
    //                               (int**)x, (int**)kpMats, (int**)&result,
    //                               M, N, K, KP_MAT_N, KP_MAT_K, stream));
  } else if (std::is_same<T, double>::value) {
    result[0] = NULL;
  } else {
    printf("Invalid type\n");
    return;
  }

  return;
#endif
}
#endif

static fastKronError backendMalloc(fastKronBackend backend, void** ptr, size_t sz) {
  switch(backend) {
    case fastKronBackend_CUDA:
      CUDACHECK(cudaMalloc(ptr, sz)); return fastKronSuccess;
    case fastKronBackend_HIP:
      HIPCHECK(hipMalloc(ptr, sz));   return fastKronSuccess;
    case fastKronBackend_ARM:
    case fastKronBackend_X86:
      {
        *ptr = (void*)(new char[sz]);
        if (*ptr == nullptr) return fastKronSuccess;
        return fastKronSuccess;
      }
    default:
      return fastKronInvalidArgument;
  }
  return fastKronSuccess;
}

static fastKronError backendFree(fastKronBackend backend, void* ptr) {
  switch(backend) {
    case fastKronBackend_CUDA:
      CUDACHECK(cudaFree(ptr)); return fastKronSuccess;
    case fastKronBackend_HIP:
      HIPCHECK(hipFree(ptr));   return fastKronSuccess;
    case fastKronBackend_ARM:
    case fastKronBackend_X86:
      delete[] (char*)ptr;
      return fastKronSuccess;
    default:
      return fastKronInvalidArgument;
  }
  return fastKronSuccess;
}

static fastKronError backendMemset(fastKronBackend backend, void* ptr, char value, size_t sz) {
  switch(backend) {
    case fastKronBackend_CUDA:
      CUDACHECK(cudaMemset(ptr, sz, value)); return fastKronSuccess;
    case fastKronBackend_HIP:
      HIPCHECK(hipMemset(ptr, sz, value));   return fastKronSuccess;
    case fastKronBackend_ARM:
    case fastKronBackend_X86:
      memset(ptr, sz, value);
      return fastKronSuccess;
    default:
      return fastKronInvalidArgument;
  }
  return fastKronSuccess;
}

static fastKronError backendMemcpyHostToDevice(fastKronBackend backend, void* dst, void* src, size_t sz) {
  switch(backend) {
    case fastKronBackend_CUDA:
      CUDACHECK(cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice)); return fastKronSuccess;
    case fastKronBackend_HIP:
      HIPCHECK(hipMemcpy(dst, src, sz, hipMemcpyHostToDevice));   return fastKronSuccess;
    case fastKronBackend_ARM:
    case fastKronBackend_X86:
      memcpy(dst, src, sz);
      return fastKronSuccess;
    default:
      return fastKronInvalidArgument;
  }
  return fastKronSuccess;
}

static fastKronError backendMemcpyDeviceToHost(fastKronBackend backend, void* dst, void* src, size_t sz) {
  switch(backend) {
    case fastKronBackend_CUDA:
      CUDACHECK(cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost)); return fastKronSuccess;
    case fastKronBackend_HIP:
      HIPCHECK(hipMemcpy(dst, src, sz, hipMemcpyDeviceToHost));   return fastKronSuccess;
    case fastKronBackend_ARM:
    case fastKronBackend_X86:
      memcpy(dst, src, sz);
      return fastKronSuccess;
    default:
      return fastKronInvalidArgument;
  }
  return fastKronSuccess;
}

/**************************************************
              Test Driver
***************************************************/
template<typename T>
static inline bool run(FastKronMMType kronmatmulType, const uint M, const uint N, const uint K, const uint NUM_KP_MATS, 
                       uint* KP_MAT_N, uint* KP_MAT_K,
                      fastKronOp opx, fastKronOp opfs,
                      uint32_t batchCountZ, uint32_t batchCountX, uint32_t batchCountF,
                      uint32_t batchCountY,
                      T alpha, T beta,
                      uint numIters, uint warmup, 
                      bool useUVA, int gpuInRows, int gpuInCols, int gpus,
                      uint kronBatch, bool checkResults, bool useFusion,
                      bool tune, fastKronBackend backend, bool verbose) {
  verbose = true;
  if (verbose)
    printf("Matmul: %d x %d x %d, Num KP Factors: %d\n", M, N, K, NUM_KP_MATS);
  bool useDistributed = gpus > 1;
  // if (useDistributed and gpuInRows * gpuInCols != gpus)
  //   printf("gpuInRows * gpuInCols != gpus: %d != %d\n", gpuInRows * gpuInCols, gpus);
  (void)useUVA;
#if !defined(ENABLE_MULTI_GPU)
  (void)gpuInCols;
  (void)gpuInRows;
  (void)gpus;
  (void)kronBatch;
#endif

  if (((batchCountZ == batchCountX && batchCountX == batchCountF) ||
      (batchCountZ == batchCountX && batchCountF == 1) ||
      (batchCountZ == batchCountF && batchCountX == 1)) &&
      (batchCountY == batchCountZ || batchCountY == 1)) {
  } else {
    printf("Wrong values for batchCountZ %d batchCountX %d  batchCountF %d\n",
           batchCountZ, batchCountX, batchCountF);
    return false;
  }

#ifdef TEST_BACKEND_CUDA
  cudaStream_t stream[gpus];
  if (backend == fastKronBackend_CUDA) {
    for (int g = 0; g < gpus; g++) {
      CUDACHECK(cudaSetDevice(g));
      CUDACHECK(cudaStreamCreate(&stream[g]));
    }
  }
#endif

#ifdef TEST_BACKEND_HIP
  hipStream_t stream[gpus];
  if (backend == fastKronBackend_HIP) {
    for (int g = 0; g < gpus; g++) {
      HIPCHECK(hipSetDevice(g));
      HIPCHECK(hipStreamCreate(&stream[g]));
    }
  }
#endif

  const uint64_t strideX = batchCountX > 1 ? M*K : 0;
  uint L = 1;
  for (int i = 0; i < NUM_KP_MATS; i++) {
    if (opfs == fastKronOp_T) {
      L = L * KP_MAT_K[i];
    } else {
      L = L * KP_MAT_N[i];
    }
  }

  const uint64_t strideZ = batchCountZ > 1 ? M*L : 0;
  const uint64_t strideY = batchCountY > 1 ? M*L : 0;
  uint64_t strideF[NUM_KP_MATS];
  
  //Allocate host data
  T* hX;
  T* hY;
  T* hKpMats[NUM_KP_MATS];
  T* hKpMatmulResult[NUM_KP_MATS];
  hX = new T[batchCountX * ((uint64_t)M) * ((uint64_t)K)];
  hY = new T[batchCountZ * ((uint64_t)M) * ((uint64_t)N)];

  for (uint i = 0; i < NUM_KP_MATS; i++) {
    hKpMats[i] = new T[batchCountF * KP_MAT_K[i] * KP_MAT_N[i]];
    strideF[i] = batchCountF > 1 ? KP_MAT_K[i] * KP_MAT_N[i] : 0;
  }

  if (verbose) printf("setting values on host\n");
  if (checkResults)
    setValues(NUM_KP_MATS, hKpMats, hX, hY, M, N, K, KP_MAT_N, KP_MAT_K, 
              batchCountX, batchCountF, batchCountY, one, one);
  if (verbose) printf("values set\n");
  printf("Supported backends %d\n", fastKronGetBackends());
  printf("FastKron %s\n", fastKronVersion());
  //Allocate GPU data
  fastKronHandle handle;
  if (verbose) printf("allocating\n");
  FastKronCHECK(fastKronInit(&handle, backend));
  uint32_t options = 0;
  if (useFusion) options = options | fastKronOptionsUseFusion;
  if (tune) options = options | fastKronOptionsTune;
  fastKronSetOptions(handle, options);

  switch (backend) {
    case fastKronBackend_CUDA:
      #ifdef TEST_BACKEND_CUDA
        if (gpus == 1)
          FastKronCHECK(fastKronInitCUDA(handle, &stream[0]));
        else {
          #ifdef ENABLE_MULTI_GPU
            FastKronCHECK(fastKronMgInitCUDA(handle, &stream[0], gpus, gpuInRows, gpuInCols, kronBatch));
          #endif
          }
      #endif
      break;
    case fastKronBackend_X86:
      FastKronCHECK(fastKronInitX86(handle));
      break;
    case fastKronBackend_HIP:
      #ifdef TEST_BACKEND_HIP
        FastKronCHECK(fastKronInitHIP(handle, &stream[0]));
      #endif
      break;
    default:
      exit(EXIT_SUCCESS);
  }
  size_t resultSize = 0;
  size_t tempSize = 0;
  FastKronCHECK(gekmmSizes(handle, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N,
                           &resultSize, &tempSize));
  resultSize = resultSize * sizeof(T);
  tempSize = tempSize * sizeof(T);
  T* dX[gpus];
  T* dY[gpus];
  T* dResult[gpus];
  T* dKpMats[gpus*NUM_KP_MATS];
  T* dTemp1[gpus];
  T *dTemp2[gpus];
  for (int i =0; i < gpus; i++) {dTemp1[i] = dTemp2[i] = nullptr;}
  uint64_t sizeX = ((uint64_t)M) * ((uint64_t)K) * sizeof(T);
#ifdef ENABLE_MULTI_GPU
  if (useDistributed) {
    FastKronCHECK(fastKronMgAllocX(handle, (void**)dX, (void**)hX, M, K));
    FastKronCHECK(fastKronMgAllocX(handle, (void**)dY, (void**)hY, M, N));
  } else
#endif
  {
    FastKronCHECK(backendMalloc(backend, (void**)&dX[0], batchCountX*sizeX));
    FastKronCHECK(backendMalloc(backend, (void**)&dY[0], batchCountY*resultSize));
  }

  if (verbose) printf("allocated\n");
  
  for (uint i = 0; i < NUM_KP_MATS; i++) {
    //TODO: Add alloc distributed for KpMats
    if (useDistributed) {
      for (int g = 0; g < gpus; g++) {
        if (backend == fastKronBackend_CUDA) CUDACHECK(cudaSetDevice(g));
        FastKronCHECK(backendMalloc(backend, (void**)&dKpMats[g * NUM_KP_MATS + i], KP_MAT_K[i] * KP_MAT_N[i] * sizeof(T)));
      }
    } else {
      FastKronCHECK(backendMalloc(backend, (void**)&dKpMats[i], batchCountF * KP_MAT_K[i] * KP_MAT_N[i] * sizeof(T)));
    }
    for (int g = 0; g < gpus; g++) {  
      FastKronCHECK(backendMemcpyHostToDevice(backend, dKpMats[g * NUM_KP_MATS + i], hKpMats[i], batchCountF * KP_MAT_K[i] * KP_MAT_N[i] * sizeof(T)));
    }
  }
  if (verbose) printf("memcpy\n");
  // if (tune) {
  //   if (std::is_same<T, float>::value)
  //     FastKronCHECK(sgekmmTune(handle, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N, opx, opfs));
  //   else if (std::is_same<T, int>::value)
  //     FastKronCHECK(igekmmTune(handle, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N, opx, opfs));
  //   else if (std::is_same<T, double>::value)
  //     FastKronCHECK(dgekmmTune(handle, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N, opx, opfs));
  //   else
  //     abort();
  // }
  printf("resultSize %lu tempSize %lu\n", resultSize, tempSize);
  for (int g = 0; g < gpus; g++) {
    if (backend == fastKronBackend_CUDA) CUDACHECK(cudaSetDevice(g));
    if (backend == fastKronBackend_HIP) HIPCHECK(hipSetDevice(g));
    FastKronCHECK(backendMalloc(backend, (void**)&dTemp1[g], batchCountZ*tempSize));
    if (resultSize < tempSize) FastKronCHECK(backendMalloc(backend, (void**)&dTemp2[g], batchCountZ*tempSize));
    FastKronCHECK(backendMalloc(backend, (void**)&dResult[g], batchCountZ*resultSize));
    FastKronCHECK(backendMemset(backend, (void*)dResult[g], 0, batchCountZ*resultSize));
  }
  printf("520 sizeX %lu\n", sizeX);
  if (checkResults) {
    if (useDistributed) {
      //Already done by allocDistributedX
    } else {
      FastKronCHECK(backendMemcpyHostToDevice(backend, dX[0], hX, batchCountX*sizeX));
      FastKronCHECK(backendMemcpyHostToDevice(backend, dY[0], hY, batchCountY*resultSize));
    }
  }
  if (verbose) printf("checkResults %d\n", checkResults);
  printf("529\n");
  if (checkResults) {
    T* hResult;
    {
      //CPU implementation of algorithm
      for (uint i = 0; i < NUM_KP_MATS; i++) {
        hKpMatmulResult[i] = new T[batchCountZ*tempSize * gpus];
      }

      slicedMatmul(kronmatmulType, NUM_KP_MATS, hKpMatmulResult, hX, hKpMats, hY, M, N, K, KP_MAT_N, KP_MAT_K, strideX, strideZ, strideF, strideY, batchCountZ, opx, opfs, alpha, beta);
      hResult = hKpMatmulResult[NUM_KP_MATS-1];
    }
    printf("540\n");
    if (verbose) printf("running kron gemm\n");
    //Run GPU implementation
#if defined(TEST_BACKEND_CUDA) && defined(ENABLE_MULTI_GPU)
    if (useDistributed) {
      kronDistributedGEMM<T>(handle, NUM_KP_MATS, dX, dKpMats, dResult, M, N, K, KP_MAT_N, KP_MAT_K, dTemp1, dTemp2, stream);
    } else 
#endif
    {
      printf("546: %p %p %p %p\n", dX[0], dY[0], dResult[0], dTemp1[0]);
      kronGEMM<T>(handle, backend, kronmatmulType, NUM_KP_MATS, dX[0], opx, dKpMats, opfs, dY[0], dResult[0], alpha, beta, M, N, K, KP_MAT_N, KP_MAT_K, batchCountZ, strideX, strideY, strideF, strideZ, dTemp1[0], dTemp2[0]);
    }
    for (int g = 0; g < gpus; g++) {
      if (backend == fastKronBackend_CUDA) { 
        CUDACHECK(cudaSetDevice(g));
        CUDACHECK(cudaDeviceSynchronize());
      } else if (backend == fastKronBackend_HIP) {
        HIPCHECK(hipSetDevice(g));
        HIPCHECK(hipDeviceSynchronize());
      }
    }
    if (verbose) printf("checking results\n");
    size_t sizeResult = batchCountZ * ((uint64_t)M) * ((uint64_t)N) * sizeof(T);
    T* dResultToHost = (T*)malloc(sizeResult);
#ifdef ENABLE_MULTI_GPU
    if (useDistributed) {
      FastKronCHECK(fastKronMgGatherY(handle, (void**)dResult, (void**)dResultToHost, M, K, NUM_KP_MATS, KP_MAT_N, KP_MAT_K));
    } else
#endif
    {
      FastKronCHECK(backendMemcpyDeviceToHost(backend, dResultToHost, dResult[0], sizeResult));
    }

    //Check Results
    if (check(hResult, dResultToHost, batchCountZ, M, N)) {
      if (verbose) printf("Results Correct\n");
    }
    else
      return false;
  }

#ifdef TEST_BACKEND_CUDA
  cudaEvent_t start[gpus];
  cudaEvent_t end[gpus];
#endif

#ifdef TEST_BACKEND_HIP
  hipEvent_t start[gpus];
  hipEvent_t end[gpus];
#endif

  if (numIters > 0 || warmup > 0) {
    if (backend == fastKronBackend_CUDA) {
      for (int g = 0; g < gpus; g++) {
        CUDACHECK(cudaSetDevice(g));
        CUDACHECK(cudaEventCreate(&start[g]));
        CUDACHECK(cudaEventCreate(&end[g]));
      }
    } else if (backend == fastKronBackend_HIP) {
      for (int g = 0; g < gpus; g++) {
        HIPCHECK(hipSetDevice(g));
        HIPCHECK(hipEventCreate(&start[g]));
        HIPCHECK(hipEventCreate(&end[g]));
      }
    }
    printf("warmup\n");
    //Warm Up iterations
    for (uint i = 0; i < warmup; i++) {
      if (useDistributed) {
#if defined(TEST_BACKEND_CUDA) && defined(ENABLE_MULTI_GPU)
        kronDistributedGEMM<T>(handle, NUM_KP_MATS, dX, dKpMats, dResult, M, N, K, KP_MAT_N, KP_MAT_K, dTemp1, dTemp2, stream);
#endif
      } else {
        kronGEMM<T>(handle, backend, kronmatmulType, NUM_KP_MATS, dX[0], opx, dKpMats, opfs, dY[0], dResult[0], alpha, beta, M, N, K, KP_MAT_N, KP_MAT_K, batchCountZ, strideX, strideY, strideF, strideZ, dTemp1[0], dTemp2[0]);
      }
    }
    if (backend == fastKronBackend_CUDA) {
      for (int g = 0; g < gpus; g++) {
        CUDACHECK(cudaSetDevice(g));
        CUDACHECK(cudaStreamSynchronize(stream[g]));
      }
    } else if (backend == fastKronBackend_HIP) {
      for (int g = 0; g < gpus; g++) {
        HIPCHECK(hipSetDevice(g));
        HIPCHECK(hipStreamSynchronize(stream[g]));
      }
    }
    printf("390\n");
    //Run
    double starttime = 0.0f;
    if (verbose) printf("run\n");
    uint32_t l3CacheSize = 384*1024*1024;
    float* cpuL3Trash1 = new float[l3CacheSize/4];
    float* cpuL3Trash2 = new float[l3CacheSize/4];
    if (backend == fastKronBackend_X86)
      setMatrix(cpuL3Trash1, l3CacheSize/4, 1, setToI);
    
    float elapsedTime = 1e10;
    for (int sample = 0; sample < 5; sample++) {
      if (backend == fastKronBackend_CUDA) {
        for (int g = 0; g < gpus; g++) {
          CUDACHECK(cudaSetDevice(g));
          CUDACHECK(cudaEventRecord(start[g], stream[g]));
        }
      } else if (backend == fastKronBackend_HIP) {
        for (int g = 0; g < gpus; g++) {
          HIPCHECK(hipSetDevice(g));
          HIPCHECK(hipEventRecord(start[g], stream[g]));
        }
      }
      float iterTime = 0.0f;
      for (uint i = 0; i < numIters; i++) {
        //printf("iter i %d\n", i);
        if (backend == fastKronBackend_X86) 
          memcpy(cpuL3Trash2, cpuL3Trash1, l3CacheSize);
        if (backend == fastKronBackend_X86) {
          starttime = getCurrTime();
        }
        if (useDistributed) {
#if defined(TEST_BACKEND_CUDA) && defined(ENABLE_MULTI_GPU)
          kronDistributedGEMM<T>(handle, NUM_KP_MATS, dX, dKpMats, dResult, M, N, K, KP_MAT_N, KP_MAT_K, dTemp1, dTemp2, stream);
#endif
        } else {
          kronGEMM<T>(handle, backend, kronmatmulType, NUM_KP_MATS, dX[0], opx, dKpMats, opfs, dY[0], dResult[0], alpha, beta, M, N, K, KP_MAT_N, KP_MAT_K, batchCountZ, strideX, strideY, strideF, strideZ, dTemp1[0], dTemp2[0]);
        }
        if (backend == fastKronBackend_X86) {
          double endtime = getCurrTime();
          iterTime += (float)(endtime - starttime)/1000.0f;
          // printf("elapsedTime %f starttime %f endtime %f\n", elapsedTime, starttime, endtime);
        }
      }
      printf("405\n");
      if (backend == fastKronBackend_CUDA) {
        for (int g = 0; g < gpus; g++) {
          CUDACHECK(cudaSetDevice(g));
          CUDACHECK(cudaEventRecord(end[g], stream[g]));
          CUDACHECK(cudaEventSynchronize(end[g]));
          if (g == 0) {
            float t = std::numeric_limits<float>::max();
            CUDACHECK(cudaEventElapsedTime(&t, start[g], end[g]));
            elapsedTime = std::min(elapsedTime, t);
          }
        }
      } else if (backend == fastKronBackend_HIP) {
        for (int g = 0; g < gpus; g++) {
          HIPCHECK(hipSetDevice(g));
          HIPCHECK(hipEventRecord(end[g], stream[g]));
          HIPCHECK(hipEventSynchronize(end[g]));
          if (g == 0) {
            float t = std::numeric_limits<float>::max();
            HIPCHECK(hipEventElapsedTime(&t, start[g], end[g]));
            elapsedTime = std::min(elapsedTime, t);
          }
        }
      } else if (backend == fastKronBackend_X86) {
        // double endtime = getCurrTime();
        elapsedTime = std::min(elapsedTime, iterTime);
        printf("elapsedTime %f\n", elapsedTime);
      }
    }

    delete[] cpuL3Trash1;
    delete[] cpuL3Trash2;

    double perCallTime = elapsedTime/numIters;
    size_t operations = 0;
    long tmpK = K;
    for (int i = NUM_KP_MATS - 1; i >= 0; i--) {
      tmpK = (tmpK/KP_MAT_K[i]) * KP_MAT_N[i];
      operations += tmpK * KP_MAT_K[i];
    }
    //Add for Alpha and Beta
    operations += N + ((beta != 0) ? 2*N : beta);
    operations = 2 * ((long)M) * operations;
    operations = batchCountZ * operations;
    double flops = operations/perCallTime;
    double gFLOPS = flops/1e9*1e3;
    printf("Time: %f ms; Operations: %ld; GFLOPS: %lf \n", perCallTime, operations, gFLOPS);
  }

  //Free GPU Memory
  for (int g = 0; g < gpus; g++) {
    if (backend == fastKronBackend_CUDA) CUDACHECK(cudaSetDevice(g));
    if (backend == fastKronBackend_HIP) HIPCHECK(hipSetDevice(g));
    FastKronCHECK(backendFree(backend, dX[g]));
    FastKronCHECK(backendFree(backend, dY[g]));
    for (uint i = 0; i < NUM_KP_MATS; i++) {
      FastKronCHECK(backendFree(backend, dKpMats[g * NUM_KP_MATS + i]));
    }
    FastKronCHECK(backendFree(backend, dTemp1[g]));
    FastKronCHECK(backendFree(backend, dTemp2[g]));
    FastKronCHECK(backendFree(backend, dResult[g]));
  }

  fastKronDestroy(handle);
  
  //Free CPU RAM
  delete[] hX;
  delete[] hY;
  for (uint i = 0; i < NUM_KP_MATS; i++) {
    delete[] hKpMats[i];
    if (checkResults) delete[] hKpMatmulResult[i];
  }

  return true;
}

#endif
