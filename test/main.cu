#include <iostream>
#include <string>

#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include "kron.h"
#include "anyoption.h"

#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Failed: Cuda error %s:%d '%s'\n",             \
          __FILE__,__LINE__,cudaGetErrorString(e));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)

/**************************************************
                    Timing functions
**************************************************/
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

/**************************************************
                Matrix Functions
***************************************************/
int one(int i, int j) {return 1;}
int zeroOne(int i, int j) {return i % 2;}
int setToI(int i, int j) {return i;}
int randMod(int i, int j) {return rand()%5 + 1;}

template<typename T>
static void setMatrix(T* mat, int M, int N, int (*fnvalue)(int i, int j)) {
  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      mat[i*N + j] = (T)fnvalue(i,j);
    }
  }
}

template<typename T>
void setValues(int NUM_KP_MATS, T* kpMats[], T *x, int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[], int (*fnvalue)(int i, int j))
{
  for (int i = 0; i < NUM_KP_MATS; i++) {
    setMatrix(kpMats[i], KP_MAT_K[i], KP_MAT_N[i], fnvalue);
  }

  setMatrix(x, M, K, fnvalue);
}

// static void printMatrix(int* mat, int M, int N, int max_rows = -1, int max_cols = -1) {
//   printf("[");
//   for (int i = 0; i < M; i++) {
//     for (int j = 0; j < N; j++) {
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
template<typename T> static bool eqVal(T x, T y) {abort(); return false;}

template<> bool eqVal(int x, int y) {return x == y;}

template<> bool eqVal(float x, float y) {
  if (abs(x) <= 1e-5 && abs(y) <= 1e-5) return true;
  if (abs(y) <= 1e-5) return abs((x-y)/x) <= 1e-5;
  return abs((x-y)/y) <= 1e-5;
}

template<typename T>
static bool check(T* ref, T* computed, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (!eqVal(ref[i*N + j], computed[i* N + j])) {
        std::cout << "Mismatch for" << M << " x " << N << " at (" << i << ", " << j << "): ref = " << ref[i*N+j] << "computed = " << computed[i*N+j] << "\n";
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
void baselineKPThenMatmul(int NUM_KP_MATS, int* result, int* x, int* kpout[], int* kpMats[],
                          int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[]) {
  int cols;
  int rows;

  for (int kp = 0; kp < NUM_KP_MATS - 1; kp++) {
    int* kpFirst = (kp == 0) ? kpMats[0] : kpout[kp - 1];
    int kpFirstRows = (kp == 0) ? KP_MAT_K[0] : rows;
    int kpFirstCols = (kp == 0) ? KP_MAT_N[0] : cols;

    cols = kpFirstCols * KP_MAT_N[kp+1];
    rows = kpFirstRows * KP_MAT_K[kp+1];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        int v2 = kpMats[kp+1][(i%KP_MAT_K[kp+1]) * KP_MAT_N[kp+1] + j%KP_MAT_N[kp+1]];
        int v1 = kpFirst[(i/KP_MAT_K[kp+1]) * kpFirstCols + j/KP_MAT_N[kp+1]];
        kpout[kp][i*cols + j] = v1 * v2;
      }
    }
  }

  for(int i = 0; i < M; i++) {    
    for(int j = 0; j < N; j++) {    
      result[i* N + j] = 0;    
      for(int k = 0; k < K; k++) {   
        result[i * N + j] += x[i*K + k]*kpout[NUM_KP_MATS-2][k*N + j];
      }    
    }    
  }

  // printMatrix(result, M, N, 4, 4);
}

//Serial implementation of the new Kron GEMM implementation
template<typename T>
void slicedMatmul(int NUM_KP_MATS, T* kpMatmulResult[], T* x, T* kpMats[],
                  int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[]) {
  int secFacRowMulSize = 1;
  int rowsTillNow = 1;
  int colsTillNow = 1;
  int resultCols;
  for (int kp = 0; kp < NUM_KP_MATS; kp++) {
    T* prevKPMatmul = (kp == 0) ? x : kpMatmulResult[kp - 1];
    int kpSecondK = KP_MAT_K[NUM_KP_MATS - 1 - kp];
    int kpSecondN = KP_MAT_N[NUM_KP_MATS - 1 - kp];
    int prevKPMatmulCols = (kp == 0) ? K : resultCols;

    resultCols = (prevKPMatmulCols/kpSecondK) * kpSecondN;
    secFacRowMulSize = (kp == 0) ? K/kpSecondK : rowsTillNow * (K/(colsTillNow * KP_MAT_K[NUM_KP_MATS - 1 - (kp)]));
    //Number of times a column is multiplied with input matrix is equal to 
    //N/(number of column elements of this matrix * cols so far) * number of rows so far.
    rowsTillNow *= KP_MAT_N[NUM_KP_MATS - 1 - (kp)];
    colsTillNow *= KP_MAT_K[NUM_KP_MATS - 1 - (kp)];

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < resultCols; j++) {
        int r = 0;

        for (int kp_k = 0; kp_k < kpSecondK; kp_k++) {
          int slice = (j / secFacRowMulSize) % kpSecondN;

          int v2 = kpMats[NUM_KP_MATS - 1 - kp][kp_k*kpSecondN + slice];
          
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
static T* kronGEMM(const int NUM_KP_MATS, T* kpMatmulResult[], T* x, T* kpMats[],
            int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[], cudaStream_t stream) {
  T* result;
  if (std::is_same<T, float>::value) {
    CUDACHECK(kronSGEMM(NUM_KP_MATS, 
                        (float**)kpMatmulResult, (float*)x, (float**)kpMats, (float**)&result,
                        M, N, K, KP_MAT_N, KP_MAT_K, stream));
  } else if (std::is_same<T, int>::value) {
    CUDACHECK(kronIGEMM(NUM_KP_MATS, 
                        (int**)kpMatmulResult, (int*)x, (int**)kpMats, (int**)&result, 
                        M, N, K, KP_MAT_N, KP_MAT_K, stream));
  } else {
    printf("Invalid type\n");
    return NULL;
  }

  return result;
}


/**************************************************
              Test Driver
***************************************************/
template<typename T, typename VecT>
static bool run(const int M, const int N, const int K, const int NUM_KP_MATS, 
                int* KP_MAT_N, int* KP_MAT_K, int numIters, bool checkResults) {
  printf("Matmul: %d x %d x %d, Num KP Factors: %d\n", M, N, K, NUM_KP_MATS);

  //Allocate host data
  T* hX;
  T* hKpMats[NUM_KP_MATS];
  T* hKpMatmulResult[NUM_KP_MATS];

  hX = new T[M*K];
  for (int i = 0; i < NUM_KP_MATS; i++) {
    hKpMats[i] = new T[KP_MAT_K[i] * KP_MAT_N[i]];
    hKpMatmulResult[i] = new T[M*std::max(N,K)];
  }

  setValues(NUM_KP_MATS, hKpMats, hX, M, N, K, KP_MAT_N, KP_MAT_K, randMod);

  //Allocate GPU data
  T* dX;
  T* dKpMatmulResult[2];
  T* dKpMats[NUM_KP_MATS];

  CUDACHECK(cudaMalloc(&dX, M * K * sizeof(T)));
  CUDACHECK(cudaMalloc(&dKpMatmulResult[0], M * std::max(N,K) * sizeof(T)));
  CUDACHECK(cudaMalloc(&dKpMatmulResult[1], M * std::max(N,K) * sizeof(T)));
  
  for (int i = 0; i < NUM_KP_MATS; i++) {
    CUDACHECK(cudaMalloc(&dKpMats[i],            KP_MAT_K[i] * KP_MAT_N[i] * sizeof(T)));
    CUDACHECK(cudaMemcpy(dKpMats[i], hKpMats[i], KP_MAT_K[i] * KP_MAT_N[i] * sizeof(T), cudaMemcpyHostToDevice));
  }

  for (int i = 0; i < 2; i++) {
    CUDACHECK(cudaMemset(dKpMatmulResult[i], 0, M * std::max(N,K) * sizeof(T)));
  }

  CUDACHECK(cudaMemcpy(dX, hX, M * K * sizeof(T), cudaMemcpyHostToDevice));
  
  if (checkResults) {
    T* dResult;
    T* hResult;

    //CPU implementation of algorithm
    slicedMatmul(NUM_KP_MATS, hKpMatmulResult, hX, hKpMats, M, N, K, KP_MAT_N, KP_MAT_K);
    hResult = hKpMatmulResult[NUM_KP_MATS-1];

    //Run GPU implementation
    dResult = kronGEMM<T>(NUM_KP_MATS, dKpMatmulResult, dX, dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, 0);
    CUDACHECK(cudaDeviceSynchronize());
    T* dResultToHost = new T[M*N];
    CUDACHECK(cudaMemcpy(dResultToHost, dResult, M*N*sizeof(T), cudaMemcpyDeviceToHost));
    
    //Check Results
    if (check(hResult, dResultToHost, M, N))
      printf("Results Correct\n");
    else
      return false;
  }

  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t end;
  float elapsedTime = 0;
  
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&end));
  cudaStreamCreate(&stream);

  //Warm Up iterations
  for (int i = 0; i < 10; i++) {
    kronGEMM<T>(NUM_KP_MATS, dKpMatmulResult, dX, dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, stream);
  }
  CUDACHECK(cudaStreamSynchronize(stream));

  //Run
  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < numIters; i++)
    kronGEMM<T>(NUM_KP_MATS, dKpMatmulResult, dX, dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, stream);
  CUDACHECK(cudaEventRecord(end, stream));
  CUDACHECK(cudaEventSynchronize(end));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, end));
  printf("elapsedtime %f milliseconds\n", elapsedTime/numIters);

  //Free GPU Memory
  for (int i = 0; i < NUM_KP_MATS; i++) {
    CUDACHECK(cudaFree(dKpMats[i]));
  }

  CUDACHECK(cudaFree(dKpMatmulResult[0]));
  CUDACHECK(cudaFree(dKpMatmulResult[1]));
  CUDACHECK(cudaFree(dX));

  //Free CPU RAM
  delete[] hX;
  for (int i = 0; i < NUM_KP_MATS; i++) {
    delete[] hKpMats[i];
    delete[] hKpMatmulResult[i];
  }

  return true;
}

/**************************************************
              Main Function
***************************************************/
int main(int argc, char* argv[]) {  
  int batch = 0;
  int facs = 0;
  int size = 0;
  char* type;

  AnyOption *opt = new AnyOption();

  opt->addUsage("usage: ");
  opt->addUsage("batch: Size of Batch");
  opt->addUsage("facs: Number of Kron Factors");
  opt->addUsage("size: Row and cols of each Kron Factor");
  opt->addUsage("type: Type of matrices (float, int, half, double)");

  opt->setOption("batch", 'b');
  opt->setOption("facs", 'f');
  opt->setOption("size", 's');
  opt->setOption("type", 't');

  opt->processCommandArgs(argc, argv);
  
  if (!opt->hasOptions()) { /* print usage if no options */
    opt->printUsage();
    delete opt;
    return 1;
  }

  if (opt->getValue('b') != NULL) {
    batch = atoi(opt->getValue('b'));
  }

  if (opt->getValue('f') != NULL) {
    facs = atoi(opt->getValue('f'));
  }

  if (opt->getValue('s') != NULL) {
    size = atoi(opt->getValue('s'));
  }

  if (opt->getValue('t') != NULL) {
    type = opt->getValue('t');
  }

  if (batch <= 0 || facs <= 0 || size <= 0) {
    printf("Invalid value batch: %d, facs %d, size %d\n", batch, facs, size);
    return 1;
  }

  int KP_MAT_N[facs];
  int KP_MAT_K[facs];
  int N = 1;
  int K = 1;
  for (int i = 0; i < facs; i++) {
    N *= size;
    K *= size;
    KP_MAT_K[i] = KP_MAT_N[i] = size;
  }
  
  bool status = false;
  if (strcmp(type, "float") == 0)
    status = run<float, float4>(batch, N, K, facs, KP_MAT_N, KP_MAT_K, 100, true);
  else if (strcmp(type, "int") == 0)
    status = run<int, int4>(batch, N, K, facs, KP_MAT_N, KP_MAT_K, 100, true);
  else
    printf("type not supported %s\n", type);

  if (!status) return 1;

  return 0;
}
