#include <iostream>
#include <string>

#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include "fastkron.h"
#include "handle/handle.h"

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
int one(int i, int j) {return 1;}
int zeroOne(int i, int j) {return i % 2;}
int setToI(int i, int j) {return i;}
int setToJ(int i, int j) {return j;}
int iPlusJ(int i, int j) {return i + j;}
int randMod(int i, int j) {return rand()%3 + 1;}

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

template<typename T>
T* transpose(uint M, uint N, T* data) {
  T* Trdata = new T[M * N];

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      Trdata[n*M + m] = data[m*N + n];
    }
  }

  return Trdata;
}

void swap(uint& X, uint& Y) {
  uint Temp = X;
  X = Y;
  Y = Temp;
}

//Serial implementation of the new Kron GEMM implementation
template<typename T>
void slicedMatmul(uint NUM_KP_MATS, T* kpMatmulResult[], T* x, T* kpMats[],
                  uint M, uint N, uint K, uint KP_MAT_N[], uint KP_MAT_K[],
                  fastKronOp opx, fastKronOp opfs) {
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
            v2 = kpMats[NUM_KP_MATS - 1 - kp][slice*KP_MAT_K[NUM_KP_MATS - 1 - kp] + kp_k];
          } else {
            v2 = kpMats[NUM_KP_MATS - 1 - kp][kp_k*kpSecondN + slice];
          }

          T v1;
          if (opx == fastKronOp_T && kp == 0)
            v1 = prevKPMatmul[((j*kpSecondK)%prevKPMatmulCols + kp_k) * M + i];
          else
            v1 = prevKPMatmul[i* prevKPMatmulCols + (j*kpSecondK)%prevKPMatmulCols + kp_k];
          r += v1 * v2;
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
static void kronGEMM(fastKronHandle handle, const uint NUM_KP_MATS, T* x, fastKronOp opx, T* kpMats[], fastKronOp opfs, T* result,
                     uint M, uint N, uint K, uint KP_MAT_N[], uint KP_MAT_K[], T* temp1, T* temp2) {
  if (std::is_same<T, float>::value) {
    CUDACHECK(sgekmm(handle, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N,  
                     (float*)x, opx, (float**)kpMats, opfs, (float*)result,
                     1, 0, nullptr, (float*)temp1, (float*)temp2));
  } else if (std::is_same<T, int>::value) {
    CUDACHECK(igekmm(handle, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N,  
                     (int*)x, opx, (int**)kpMats, opfs, (int*)result,
                     1, 0, nullptr, (int*)temp1, (int*)temp2));
  } else if (std::is_same<T, double>::value) {
    CUDACHECK(dgekmm(handle, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N,  
                     (double*)x, opx, (double**)kpMats, opfs, (double*)result,
                     1, 0, nullptr, (double*)temp1, (double*)temp2));
  } else {
    printf("Invalid type\n");
    return;
  }

  return;
}

template<typename T>
static void kronDistributedGEMM(fastKronHandle handle, const uint NUM_KP_MATS, T* x[], T* kpMats[], T* result[],
            uint M, uint N, uint K, uint KP_MAT_N[], uint KP_MAT_K[], 
            T* temp1[], T* temp2[], cudaStream_t stream[]) {
  if (std::is_same<T, float>::value) {
    CUDACHECK(kronDistributedSGEMM(handle, NUM_KP_MATS,
                                  (float**)x, (float**)kpMats, (float**)result,
                                  M, N, K, KP_MAT_N, KP_MAT_K, 
                                  (float**)temp1, (float**)temp2, 
                                  stream));
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
}

static cudaError_t backendMalloc(fastKronBackend backend, void** ptr, size_t sz) {
  switch(backend) {
    case fastKronBackend_CUDA:
      return cudaMalloc(ptr, sz);
    case fastKronBackend_ARM:
    case fastKronBackend_X86:
      {
        *ptr = (void*)(new char[sz]);
        if (*ptr == nullptr) return cudaSuccess;
        return cudaSuccess;
      }
  }
  return cudaSuccess;
}

static cudaError_t backendFree(fastKronBackend backend, void* ptr) {
  switch(backend) {
    case fastKronBackend_CUDA:
      return cudaFree(ptr);
    case fastKronBackend_ARM:
    case fastKronBackend_X86:
      delete ptr;
      return cudaSuccess;
  }
  return cudaSuccess;
}

static cudaError_t backendMemset(fastKronBackend backend, void* ptr, size_t sz, char value) {
  switch(backend) {
    case fastKronBackend_CUDA:
      return cudaMemset(ptr, sz, value);
    case fastKronBackend_ARM:
    case fastKronBackend_X86:
      memset(ptr, sz, value);
      return cudaSuccess;
  }
  return cudaSuccess;
}

static cudaError_t backendMemcpyHostToDevice(fastKronBackend backend, void* dst, void* src, size_t sz) {
  switch(backend) {
    case fastKronBackend_CUDA:
      return cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice);
    case fastKronBackend_ARM:
    case fastKronBackend_X86:
      memcpy(dst, src, sz);
      return cudaSuccess;
  }
  return cudaSuccess;
}

static cudaError_t backendMemcpyDeviceToHost(fastKronBackend backend, void* dst, void* src, size_t sz) {
  switch(backend) {
    case fastKronBackend_CUDA:
      return cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost);
    case fastKronBackend_ARM:
    case fastKronBackend_X86:
      memcpy(dst, src, sz);
      return cudaSuccess;
  }
  return cudaSuccess;
}

/**************************************************
              Test Driver
***************************************************/
template<typename T>
static bool run(const uint M, const uint N, const uint K, const uint NUM_KP_MATS, 
                uint* KP_MAT_N, uint* KP_MAT_K,
                fastKronOp opx, fastKronOp opfs,
                uint numIters, uint warmup, 
                bool useUVA, int gpuInRows, int gpuInCols, int gpus,
                uint kronBatch, bool checkResults, bool useFusion, bool tune, fastKronBackend backend, bool verbose) {
  verbose = true;
  if (verbose)
    printf("Matmul: %d x %d x %d, Num KP Factors: %d\n", M, N, K, NUM_KP_MATS);
  bool useDistributed = gpus > 1;
  // if (useDistributed and gpuInRows * gpuInCols != gpus)
  //   printf("gpuInRows * gpuInCols != gpus: %d != %d\n", gpuInRows * gpuInCols, gpus);
  cudaStream_t stream[gpus];
  if (backend == fastKronBackend_CUDA) {
    for (int g = 0; g < gpus; g++) {
      CUDACHECK(cudaSetDevice(g));
      cudaStreamCreate(&stream[g]);
    }
  }

  //Allocate host data
  T* hX;
  T* hKpMats[NUM_KP_MATS];
  T* hKpMatmulResult[NUM_KP_MATS];
  hX = new T[((uint64_t)M) * ((uint64_t)K)];
  for (uint i = 0; i < NUM_KP_MATS; i++) {
    hKpMats[i] = new T[KP_MAT_K[i] * KP_MAT_N[i]];
  }
  if (verbose) printf("setting values on host\n");
  if (checkResults)
    setValues(NUM_KP_MATS, hKpMats, hX, M, N, K, KP_MAT_N, KP_MAT_K, one);
  if (verbose) printf("values set\n");
  //Allocate GPU data
  fastKronHandle handle;
  if (verbose) printf("allocating\n");
  CUDA_CHECK(fastKronInit(&handle, backend));
  handle->setUseFusion(useFusion);
  if (backend == fastKronBackend_CUDA)
    CUDA_CHECK(fastKronInitCUDA(handle, &stream[0], gpus, gpuInRows, gpuInCols, kronBatch));
  else if (backend == fastKronBackend_X86)
    CUDA_CHECK(fastKronInitX86(handle));
  size_t resultSize = 0;
  size_t tempSize = 0;
  CUDACHECK(gekmmSizes(handle, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N,
                       &resultSize, &tempSize));
  T* dX[gpus];
  T* dResult[gpus];
  T* dKpMats[gpus*NUM_KP_MATS];
  T* dTemp1[gpus] = {nullptr};
  T *dTemp2[gpus] = {nullptr};
  uint64_t sizeX = ((uint64_t)M) * ((uint64_t)K) * sizeof(T);
  if (useDistributed) {
    CUDACHECK(allocDistributedX(handle, dX, hX, M, K));
  } else {
    CUDACHECK(backendMalloc(backend, (void**)&dX[0], sizeX));
  }
  
  if (verbose) printf("allocated\n");
  
  for (uint i = 0; i < NUM_KP_MATS; i++) {
    //TODO: Add alloc distributed for KpMats
    if (useDistributed) {
      for (int g = 0; g < gpus; g++) {
        CUDACHECK(cudaSetDevice(g));
        CUDACHECK(backendMalloc(backend, (void**)&dKpMats[g * NUM_KP_MATS + i], KP_MAT_K[i] * KP_MAT_N[i] * sizeof(T)));
      }
    } else {
      CUDACHECK(backendMalloc(backend, (void**)&dKpMats[i], KP_MAT_K[i] * KP_MAT_N[i] * sizeof(T)));
    }
    for (int g = 0; g < gpus; g++) {  
      CUDACHECK(backendMemcpyHostToDevice(backend, dKpMats[g * NUM_KP_MATS + i], hKpMats[i], KP_MAT_K[i] * KP_MAT_N[i] * sizeof(T)));
    }
  }
  if (verbose) printf("memcpy\n");
  if (tune) {
    CUDACHECK(sgekmmTune(handle, M, NUM_KP_MATS, KP_MAT_K, KP_MAT_N, opx, opfs));
  }
    
  for (int g = 0; g < gpus; g++) {
    CUDACHECK(cudaSetDevice(g));
    CUDACHECK(backendMalloc(backend, (void**)&dTemp1[g], tempSize));
    if (resultSize < tempSize)
      CUDACHECK(backendMalloc(backend, (void**)&dTemp2[g], tempSize));
    CUDACHECK(backendMalloc(backend, (void**)&dResult[g], resultSize));
    CUDACHECK(backendMemset(backend, (void*)dResult[g], 0, resultSize));
  }
  
  if (checkResults) {
    if (useDistributed) {
      //Already done by allocDistributedX
    } else {
      CUDACHECK(backendMemcpyHostToDevice(backend, dX[0], hX, sizeX));
    }
  }
  if (verbose) printf("checkResults %d\n", checkResults);
  if (checkResults) {
    T* hResult;
    {
      //CPU implementation of algorithm
      for (uint i = 0; i < NUM_KP_MATS; i++) {
        hKpMatmulResult[i] = new T[tempSize * gpus];
      }

      slicedMatmul(NUM_KP_MATS, hKpMatmulResult, hX, hKpMats, M, N, K, KP_MAT_N, KP_MAT_K, opx, opfs);
      hResult = hKpMatmulResult[NUM_KP_MATS-1];
    }
    if (verbose) printf("running kron gemm\n");
    //Run GPU implementation
    if (useDistributed) {
      kronDistributedGEMM<T>(handle, NUM_KP_MATS, dX, dKpMats, dResult, M, N, K, KP_MAT_N, KP_MAT_K, dTemp1, dTemp2, stream);
    } else {
      kronGEMM<T>(handle, NUM_KP_MATS, dX[0], opx, dKpMats, opfs, dResult[0], M, N, K, KP_MAT_N, KP_MAT_K, dTemp1[0], dTemp2[0]);
    }
    for (int g = 0; g < gpus; g++) {
      CUDACHECK(cudaSetDevice(g));
      CUDACHECK(cudaDeviceSynchronize());
    }
    if (verbose) printf("checking results\n");
    size_t sizeResult = ((uint64_t)M) * ((uint64_t)N) * sizeof(T);
    printf("sizeResult %ld resultSize %ld\n", sizeResult, resultSize * sizeof(T));
    T* dResultToHost = (T*)malloc(sizeResult);
    CUDACHECK(cudaDeviceSynchronize());
    if (useDistributed) {
      CUDACHECK(gatherDistributedY(handle, dResult, dResultToHost, M, K, NUM_KP_MATS, KP_MAT_N, KP_MAT_K));
    } else {
      CUDACHECK(backendMemcpyDeviceToHost(backend, dResultToHost, dResult[0], sizeResult));
    }

    //Check Results
    if (check(hResult, dResultToHost, M, N)) {
      if (verbose) printf("Results Correct\n");
    }
    else
      return false;
  }

  cudaEvent_t start[gpus];
  cudaEvent_t end[gpus];
  float elapsedTime = 1e10;

  if (numIters > 0 || warmup > 0) {
    if (backend == fastKronBackend_CUDA) {
      for (int g = 0; g < gpus; g++) {
        CUDACHECK(cudaSetDevice(g));
        CUDACHECK(cudaEventCreate(&start[g]));
        CUDACHECK(cudaEventCreate(&end[g]));
      }
    }
    printf("warmup\n");
    //Warm Up iterations
    for (uint i = 0; i < warmup; i++) {
      if (useDistributed) {
        kronDistributedGEMM<T>(handle, NUM_KP_MATS, dX, dKpMats, dResult, M, N, K, KP_MAT_N, KP_MAT_K, dTemp1, dTemp2, stream);
      } else {
        kronGEMM<T>(handle, NUM_KP_MATS, dX[0], opx, dKpMats, opfs, dResult[0], M, N, K, KP_MAT_N, KP_MAT_K, dTemp1[0], dTemp2[0]);
      }
    }
    if (backend == fastKronBackend_CUDA) {
      for (int g = 0; g < gpus; g++) {
        CUDACHECK(cudaSetDevice(g));
        CUDACHECK(cudaStreamSynchronize(stream[g]));
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
    
    for (int sample = 0; sample < 5; sample++) {
      if (backend == fastKronBackend_CUDA) {
        for (int g = 0; g < gpus; g++) {
          CUDACHECK(cudaSetDevice(g));
          CUDACHECK(cudaEventRecord(start[g], stream[g]));
        }
      }
      elapsedTime = 0.0f;
      for (uint i = 0; i < numIters; i++) {
        //printf("iter i %d\n", i);
        if (backend == fastKronBackend_X86) 
          memcpy(cpuL3Trash2, cpuL3Trash1, l3CacheSize);
        if (backend == fastKronBackend_X86) {
          starttime = getCurrTime();
        }
        if (useDistributed) {
          kronDistributedGEMM<T>(handle, NUM_KP_MATS, dX, dKpMats, dResult, M, N, K, KP_MAT_N, KP_MAT_K, dTemp1, dTemp2, stream);
        } else {
          kronGEMM<T>(handle, NUM_KP_MATS, dX[0], opx, dKpMats, opfs, dResult[0], M, N, K, KP_MAT_N, KP_MAT_K, dTemp1[0], dTemp2[0]);
        }
        if (backend == fastKronBackend_X86) {
          double endtime = getCurrTime();
          elapsedTime += (float)(endtime - starttime)/1000.0f;
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
            float t;
            CUDACHECK(cudaEventElapsedTime(&t, start[g], end[g]));
            elapsedTime = std::min(elapsedTime, t);
          }
        }
      } else if (backend == fastKronBackend_X86) {
        // double endtime = getCurrTime();
        // elapsedTime = std::min(elapsedTime, (float)(endtime - starttime)/1000.0f);
        printf("elapsedTime %f\n", elapsedTime);
      }
    }

    double perCallTime = elapsedTime/numIters;
    size_t operations = 0;
    long tmpK = K;
    for (int i = NUM_KP_MATS - 1; i >= 0; i--) {
      tmpK = (tmpK/KP_MAT_K[i]) * KP_MAT_N[i];
      operations += tmpK * KP_MAT_K[i];
    }
    operations = 2 * ((long)M) * operations;
    double flops = operations/perCallTime;
    double gFLOPS = flops/1e9*1e3;
    printf("Time: %f ms; Operations: %ld; GFLOPS: %lf \n", perCallTime, operations, gFLOPS);
  }

  //Free GPU Memory
  for (int g = 0; g < gpus; g++) {
    CUDACHECK(cudaSetDevice(g));
    CUDACHECK(backendFree(backend, dX[g]));
    for (uint i = 0; i < NUM_KP_MATS; i++) {
      CUDACHECK(backendFree(backend, dKpMats[g * NUM_KP_MATS + i]));
    }
    CUDACHECK(backendFree(backend, dTemp1[g]));
    CUDACHECK(backendFree(backend, dTemp2[g]));
  }

  fastKronDestroy(handle);
  
  //Free CPU RAM
  delete[] hX;
  for (uint i = 0; i < NUM_KP_MATS; i++) {
    delete[] hKpMats[i];
    if (checkResults) delete[] hKpMatmulResult[i];
  }

  return true;
}

#endif
