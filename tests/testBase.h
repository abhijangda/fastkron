#include <iostream>
#include <string>

#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include "kron.h"

#ifndef __TEST_BASE_H__
#define __TEST_BASE_H__

#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Failed: Cuda error %s:%d '%s'\n",             \
          __FILE__,__LINE__,cudaGetErrorString(e));   \
      abort();                             \
    }                                                 \
  } while(0)

/**************************************************
                Matrix Functions
***************************************************/
int one(int i, int j) {return 1;}
int zeroOne(int i, int j) {return i % 2;}
int setToI(int i, int j) {return i;}
int randMod(int i, int j) {return rand()%2 + 1;}

template<typename T>
static void setMatrix(T* mat, uint M, uint N, int (*fnvalue)(int i, int j)) {
  for (uint i = 0; i < M; i++) {    
    for (uint j = 0; j < N; j++) {
      mat[i*N + j] = (T)fnvalue(i,j);
    }
  }
}

template<typename T>
void setValues(uint NUM_KP_MATS, T* kpMats[], T *x, uint M, uint N, uint K, uint KP_MAT_N[], uint KP_MAT_K[], int (*fnvalue)(int i, int j))
{
  for (uint i = 0; i < NUM_KP_MATS; i++) {
    setMatrix(kpMats[i], KP_MAT_K[i], KP_MAT_N[i], fnvalue);
  }

  setMatrix(x, M, K, fnvalue);
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
template<typename T> static bool eqVal(T x, T y) {abort(); printf("invalid type\n"); return false;}

template<> bool eqVal(int x, int y) {return x == y;}

template<> bool eqVal(float x, float y) {
  if (abs(x) <= 1e-5 && abs(y) <= 1e-5) return true;
  if (abs(y) <= 1e-5) return abs((x-y)/x) <= 1e-5;
  return abs((x-y)/y) <= 1e-5;
}

template<> bool eqVal(double x, double y) {
  if (abs(x) <= 1e-5 && abs(y) <= 1e-5) return true;
  if (abs(y) <= 1e-5) return abs((x-y)/x) <= 1e-5;
  return abs((x-y)/y) <= 1e-5;
}

template<typename T>
static bool check(T* ref, T* computed, uint M, uint N) {
  for (uint i = 0; i < M; i++) {
    for (uint j = 0; j < N; j++) {
      if (!eqVal(ref[i*N + j], computed[i* N + j])) {
        std::cout << "Mismatch for " << M << " x " << N << " at (" << i << ", " << j << "): ref = " << ref[i*N+j] << " computed = " << computed[i*N+j] << "\n";
        return false;
      }
    }
  }

  return true;
}

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
void slicedMatmul(uint NUM_KP_MATS, T* kpMatmulResult[], T* x, T* kpMats[],
                  uint M, uint N, uint K, uint KP_MAT_N[], uint KP_MAT_K[]) {
  uint secFacRowMulSize = 1;
  uint rowsTillNow = 1;
  uint colsTillNow = 1;
  uint resultCols;
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

    #pragma omp parallel for
    for (uint i = 0; i < M; i++) {
      for (uint j = 0; j < resultCols; j++) {
        T r = 0;

        for (uint kp_k = 0; kp_k < kpSecondK; kp_k++) {
          uint slice = (j / secFacRowMulSize) % kpSecondN;

          T v2 = kpMats[NUM_KP_MATS - 1 - kp][kp_k*kpSecondN + slice];
          
          r += prevKPMatmul[i* prevKPMatmulCols + (j*kpSecondK)%prevKPMatmulCols + kp_k] * v2;
        }

        kpMatmulResult[kp][i*resultCols + j] = r;
      }
    }
  }
}

/**************************************************
              Call KronGEMM Library Functions
***************************************************/
template<typename T>
static T* kronGEMM(FastKronHandle& handle, const uint NUM_KP_MATS, T* x, T* kpMats[],
            uint M, uint N, uint K, uint KP_MAT_N[], uint KP_MAT_K[], cudaStream_t stream) {
  T* result;
  if (std::is_same<T, float>::value) {
    CUDACHECK(kronSGEMM(handle, NUM_KP_MATS,
                        (float*)x, (float**)kpMats, (float**)&result,
                        M, N, K, KP_MAT_N, KP_MAT_K, stream));
  } else if (std::is_same<T, int>::value) {
    CUDACHECK(kronIGEMM(handle, NUM_KP_MATS, 
                        (int*)x, (int**)kpMats, (int**)&result, 
                        M, N, K, KP_MAT_N, KP_MAT_K, stream));
  } else if (std::is_same<T, double>::value) {
    CUDACHECK(kronDGEMM(handle, NUM_KP_MATS, 
                        (double*)x, (double**)kpMats, (double**)&result,
                        M, N, K, KP_MAT_N, KP_MAT_K, stream));
  } else {
    printf("Invalid type\n");
    return NULL;
  }

  return result;
}


template<typename T>
static T* kronGEMMOutOfCore(FastKronHandle& handle, const uint NUM_KP_MATS, T* x, T* kpMats[],
            uint M, uint N, uint K, uint KP_MAT_N[], uint KP_MAT_K[], cudaStream_t stream[]) {
  T* result;
  if (std::is_same<T, float>::value) {
    CUDACHECK(kronSGEMMOutofCoreX(handle, NUM_KP_MATS,
                                  (float*)x, (float**)kpMats, (float**)&result,
                                  M, N, K, KP_MAT_N, KP_MAT_K, stream));
  } else if (std::is_same<T, int>::value) {
    CUDACHECK(kronIGEMMOutofCoreX(handle, NUM_KP_MATS,
                                  (int*)x, (int**)kpMats, (int**)&result,
                                  M, N, K, KP_MAT_N, KP_MAT_K, stream));
  } else if (std::is_same<T, double>::value) {
    result = NULL;
  } else {
    printf("Invalid type\n");
    return NULL;
  }

  return result;
}

/**************************************************
              Test Driver
***************************************************/
template<typename T>
static bool run(const uint M, const uint N, const uint K, const uint NUM_KP_MATS, 
                uint* KP_MAT_N, uint* KP_MAT_K, uint numIters, uint warmup, 
                bool useUVA, uint OnGPURows, uint MaxInnerKrons, uint NumMaxInnerKrons,
                int gpus, bool checkResults, bool useFusion, bool tune, bool verbose) {
  if (verbose)
    printf("Matmul: %d x %d x %d, Num KP Factors: %d\n", M, N, K, NUM_KP_MATS);
  cudaStream_t stream[gpus];
  for (int g = 0; g < gpus; g++) {
    CUDACHECK(cudaSetDevice(g));
    cudaStreamCreate(&stream[g]);
  }

  //Allocate host data
  T* hX;
  T* hKpMats[NUM_KP_MATS];
  T* hKpMatmulResult[NUM_KP_MATS];
  
  hX = new T[((uint64_t)M) * ((uint64_t)K)];
  for (uint i = 0; i < NUM_KP_MATS; i++) {
    hKpMats[i] = new T[KP_MAT_K[i] * KP_MAT_N[i]];
    hKpMatmulResult[i] = new T[(uint64_t)M*std::max((uint64_t)N,(uint64_t)K)];
  }
  if (verbose) printf("setting values on host\n");
  if (checkResults)
    setValues(NUM_KP_MATS, hKpMats, hX, M, N, K, KP_MAT_N, KP_MAT_K, randMod);
  if (verbose) printf("values set\n");
  //Allocate GPU data
  FastKronHandle handle(M, N, K, KP_MAT_N, KP_MAT_K, NUM_KP_MATS);
  if (verbose) printf("allocating\n");
  if (useUVA) {
    handle.setOutOfCoreRowsCols(gpus, OnGPURows, MaxInnerKrons, NumMaxInnerKrons);
    handle.init<T>(true);
  } else {
    handle.init<T>(false);
  }
  handle.setUseFusion(useFusion);
  T* dX;
  T* dKpMats[gpus*NUM_KP_MATS];

  uint64_t sizeX = ((uint64_t)M) * ((uint64_t)K) * sizeof(T);
  if (useUVA) {
    CUDACHECK(cudaMallocManaged(&dX, sizeX));  
  } else {
    CUDACHECK(cudaMalloc(&dX, sizeX));
  }
  if (verbose) printf("allocated\n");
  
  for (uint i = 0; i < NUM_KP_MATS; i++) {
    if (useUVA) {
      for (int g = 0; g < gpus; g++) {
        CUDACHECK(cudaSetDevice(g));
        CUDACHECK(cudaMalloc(&dKpMats[g * NUM_KP_MATS + i],     KP_MAT_K[i] * KP_MAT_N[i] * sizeof(T)));
      }
    } else {
      CUDACHECK(cudaMalloc(&dKpMats[i],     KP_MAT_K[i] * KP_MAT_N[i] * sizeof(T)));
    }
    for (int g = 0; g < gpus; g++) {  
      CUDACHECK(cudaMemcpy(dKpMats[g * NUM_KP_MATS + i], hKpMats[i], KP_MAT_K[i] * KP_MAT_N[i] * sizeof(T), cudaMemcpyHostToDevice));
    }
  }
  if (verbose) printf("memcpy\n");
  if (tune) {
    kronSGEMMTune(handle, NUM_KP_MATS, (float*)dX, (float**)dKpMats, M, N, K, KP_MAT_K, KP_MAT_N,
                            stream[0]);
  }
  if (checkResults)
    CUDACHECK(cudaMemcpy(dX, hX, sizeX, cudaMemcpyHostToDevice));
  if (verbose) printf("checkResults %d\n", checkResults);
  if (checkResults) {
    T* dResult;
    T* hResult;

    //CPU implementation of algorithm
    slicedMatmul(NUM_KP_MATS, hKpMatmulResult, hX, hKpMats, M, N, K, KP_MAT_N, KP_MAT_K);
    hResult = hKpMatmulResult[NUM_KP_MATS-1];
    if (verbose) printf("running kron gemm\n");
    //Run GPU implementation
    if (useUVA) {
      dResult = kronGEMMOutOfCore<T>(handle, NUM_KP_MATS, dX, dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, stream);
    } else {
      dResult = kronGEMM<T>(handle, NUM_KP_MATS, dX, dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, stream[0]);
    }
    for (int g = 0; g < gpus; g++) {
      CUDACHECK(cudaSetDevice(g));
      CUDACHECK(cudaDeviceSynchronize());
    }
    if (verbose) printf("checking results\n");
    T* dResultToHost = (T*)malloc(sizeX);
    CUDACHECK(cudaMemcpy(dResultToHost, dResult, sizeX, cudaMemcpyDeviceToHost));
    
    //Check Results
    if (check(hResult, dResultToHost, M, N)) {
      if (verbose) printf("Results Correct\n");
    }
    else
      return false;
  }

  cudaEvent_t start[gpus];
  cudaEvent_t end[gpus];
  float elapsedTime = 0;
  
  for (int g = 0; g < gpus; g++) {
    CUDACHECK(cudaSetDevice(g));
    CUDACHECK(cudaEventCreate(&start[g]));
    CUDACHECK(cudaEventCreate(&end[g]));
    cudaStreamCreate(&stream[g]);
  }
  if (verbose) printf("warmup\n");
  //Warm Up iterations
  for (uint i = 0; i < warmup; i++) {
    if (useUVA) {
      kronGEMMOutOfCore<T>(handle, NUM_KP_MATS, dX, dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, stream);
    } else {
      kronGEMM<T>(handle, NUM_KP_MATS, dX, dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, stream[0]);
    }
  }
  for (int g = 0; g < gpus; g++) {
    CUDACHECK(cudaSetDevice(g));
    CUDACHECK(cudaStreamSynchronize(stream[g]));
  }
  //Run
  if (verbose) printf("run\n");
  for (int g = 0; g < gpus; g++) {
    CUDACHECK(cudaSetDevice(g));
    CUDACHECK(cudaEventRecord(start[g], stream[g]));
  }
  for (uint i = 0; i < numIters; i++) {
    //printf("iter i %d\n", i);
    if (useUVA) {
      kronGEMMOutOfCore<T>(handle, NUM_KP_MATS, dX, dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, stream);
    } else {
      kronGEMM<T>(handle, NUM_KP_MATS, dX, dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, stream[0]);
    }
  }

  for (int g = 0; g < gpus; g++) {
    CUDACHECK(cudaSetDevice(g));
    CUDACHECK(cudaEventRecord(end[g], stream[g]));
    CUDACHECK(cudaEventSynchronize(end[g]));
    CUDACHECK(cudaEventElapsedTime(&elapsedTime, start[g], end[g]));
    printf("elapsedtime %f milliseconds\n", elapsedTime/numIters);
  }

  //Free GPU Memory
  for (int g = 0; g < gpus; g++)
    for (uint i = 0; i < NUM_KP_MATS; i++) {
      CUDACHECK(cudaFree(dKpMats[g * NUM_KP_MATS + i]));
    }

  handle.free();
  CUDACHECK(cudaFree(dX));

  //Free CPU RAM
  delete[] hX;
  for (uint i = 0; i < NUM_KP_MATS; i++) {
    delete[] hKpMats[i];
    delete[] hKpMatmulResult[i];
  }

  return true;
}

#endif