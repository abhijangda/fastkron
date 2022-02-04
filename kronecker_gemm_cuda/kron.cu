
// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#include <iostream>
#include <string>
#include <cstdlib>
#include <cassert>
#include <vector>

void setMatrix(int* mat, int M, int N, int (*fnvalue)(int i, int j)) 
{
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      mat[i*N + j] = fnvalue(i,j);
    }
  }
}

void printMatrix(int* mat, int M, int N) 
{
  printf("[");
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      // if (mat[i*N + j] == 18496)
        // printf("%d,%d\n",i,j);
      printf("%d, ", mat[i*N + j]);
    }
    if (i < M-1)
      printf("\n");
  }
  printf("]");
}

void baselineKPThenMatmul(int NUM_KP_MATS, int* result, int* x, int* kpout[], int* kpMats[],
                          int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[])
{
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
}

/**
 * 
*/
void slicedMatmul(int NUM_KP_MATS, int* kpMatmulResult[], int* x, int* kpMats[],
                  int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[])
{
  int secFacRowMulSize = 1;
  int rowsTillNow = 1;
  int colsTillNow = 1;
  int resultCols;
  for (int kp = 0; kp < NUM_KP_MATS; kp++) {
    int* prevKPMatmul = (kp == 0) ? x : kpMatmulResult[kp - 1];
    int kpSecondK = KP_MAT_K[NUM_KP_MATS - 1 - kp];
    int kpSecondN = KP_MAT_N[NUM_KP_MATS - 1 - kp];
    int prevKPMatmulCols = (kp == 0) ? K : resultCols;

    resultCols = (prevKPMatmulCols/kpSecondK) * kpSecondN;
    secFacRowMulSize = (kp == 0) ? K/kpSecondK : rowsTillNow * K/(colsTillNow * KP_MAT_K[NUM_KP_MATS - 1 - (kp)]);

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


template<typename T,int N_THREADS, int K, int N_COARSE_TB, int TILE_Y, int TILE_X, int TILE_K, int KP_N, int KP_K, int KP_K_BATCH>
__global__ 
void __launch_bounds__(N_THREADS)  cuda_gemm(int M, int N, T * A, T * kron_fac, T * C) {
  __shared__ __align__(128) int kron_fac_sh[TILE_Y][KP_K+1];//TODO: Change padding based on value o1, KP_K and TILE_Y
  __shared__ __align__(128) int Ash[TILE_X][TILE_K/KP_K][KP_K+4]; //TODO: Padding of 4 so that 128-bit loads can be loaded without misaligned address but ideally padding of 1 will be best for shared memory loads of As at line 293
  __shared__ __align__(128) int Csh[TILE_X][K];

  int wid = threadIdx.x/warpSize;
  int lane = threadIdx.x%warpSize;
  int blockWarps = blockDim.x/warpSize;

  for (auto i = threadIdx.x; i < KP_K * TILE_Y; i += blockDim.x) {
    kron_fac_sh[i%TILE_Y][i/TILE_Y] = kron_fac[(i/TILE_Y) * KP_N + blockIdx.y *TILE_Y+ (i%TILE_Y)];
  }

  typedef int4 LD_TYPE;
  const int ldNumElems = (sizeof(LD_TYPE)/sizeof(int));
  
  int tile_k = 0;
  
  for (int start_row = blockIdx.x * TILE_X; start_row < gridDim.x * TILE_X * N_COARSE_TB; start_row += gridDim.x * TILE_X) {
    for (int a_row = 0; a_row < TILE_X; a_row += 1) {
      for (int a_col = threadIdx.x*ldNumElems, ari = 0; a_col < TILE_K; a_col += blockDim.x*ldNumElems, ari++) {
        LD_TYPE a = *(LD_TYPE*)&A[(a_row + start_row) * K + (a_col + tile_k)];
        
        *(LD_TYPE*)&Ash[a_row][a_col/KP_K][a_col%KP_K] = a;

        //TODO: Use warp shuffles to avoid misaligned address and have padding of 1

        // int a1[4] = {a.x, a.y, a.z, a.w};
        //for (int round = 0; round < 4; round++)
        // __shfl_sync(0xffffffff, a[(lane+round)%4], lane/4);
      }
    }

    __syncthreads();

    for (int a_row = 0; a_row < TILE_X; a_row++) {
      int lane = threadIdx.x%KP_K;
      int wid = threadIdx.x/KP_K;
      int blockWarps = blockDim.x/KP_K; //TODO: Names should be different
      
      register int Ar[KP_K];
      
      for (int a_col = 0; a_col < KP_K; a_col++) {
        Ar[a_col] = Ash[a_row][lane][a_col]; //TODO: Specifically for KP_K=32
      }

      for (int kp_col = wid; kp_col < KP_N; kp_col += blockWarps) {
        register int kron_fac_r; //TODO: Specifically for KP_K=32 and TILE_Y=32

        kron_fac_r = kron_fac_sh[kp_col][lane];

        for (int a_col_start = lane * KP_K; a_col_start < TILE_K; a_col_start += KP_K*KP_K) {
          int c = 0;

          #pragma unroll
          for (int a_col = 0; a_col < KP_K; a_col++) {
            int a = Ash[a_row][a_col_start/KP_K][a_col]; //Ar[a_col];
            int kp_row = a_col;
            int kp = kron_fac_sh[kp_col][a_col];;//__shfl_sync(0xffffffff, kron_fac_r, a_col, KP_K);

            c += a * kp;
          }

          Csh[a_row][kp_col*(TILE_K/KP_K)+a_col_start/KP_K] = c;
        }
      }
    }

    __syncthreads();

    for (int a_row = 0; a_row < TILE_X; a_row++) {
      for (int c_col = threadIdx.x*ldNumElems; c_col < N; c_col += blockDim.x*ldNumElems) {
        int c_row = (a_row + start_row);
        int c_idx = c_row * N + c_col;
        
        *(LD_TYPE*)&C[c_idx] = *(LD_TYPE*)&Csh[a_row][c_col];
      }
    }
  }
}

void customKronGEMM(const int NUM_KP_MATS, int* kpMatmulResult[], int* x, int* kpMats[],
                     int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[], cudaStream_t stream)
{
  //Row Major Layout of all matrics
  for (int i = 0; i < NUM_KP_MATS; i++) {
    int* prev_kp = (i==0) ? x : kpMatmulResult[i-1];
    if (KP_MAT_K[0] == 8) {
      const int KP_K = 8;
      const int TILE_Y = KP_K; //Y direction corresponds to tile of column of the KP factor
      const int TILE_X = 1; //X direction correspond to tile of row 
      const int KP_K_BATCH = 1;
      const int N_COARSE_TB = 1;
      const int TILE_K = KP_K*KP_K*KP_K;
      
      dim3 grid = {M/TILE_X/N_COARSE_TB, 1};  //(N/KP_MAT_N[NUM_KP_MATS-i-1])/TILE_Y
      dim3 block = {128,1,1};
      cuda_gemm<int,128,TILE_K,N_COARSE_TB,TILE_Y,TILE_X,TILE_K,KP_K,KP_K,KP_K_BATCH><<<grid, block, 0, stream>>>(M, N, prev_kp, kpMats[NUM_KP_MATS-i-1], kpMatmulResult[i]);
    } else if (KP_MAT_K[0] == 16) {
      const int KP_K = 16;
      const int TILE_Y = KP_K; //Y direction corresponds to tile of column of the KP factor
      const int TILE_X = 1; //X direction correspond to tile of row 
      const int KP_K_BATCH = 1;
      const int N_COARSE_TB = 1;
      const int TILE_K = KP_K*KP_K;

      dim3 grid = {M/TILE_X/N_COARSE_TB, 1};  //(N/KP_MAT_N[NUM_KP_MATS-i-1])/TILE_Y
      dim3 block = {128,1,1};
      cuda_gemm<int,128,TILE_K,N_COARSE_TB,TILE_Y,TILE_X,TILE_K,KP_K,KP_K,KP_K_BATCH><<<grid, block, 0, stream>>>(M, N, prev_kp, kpMats[NUM_KP_MATS-i-1], kpMatmulResult[i]);
    } else if (KP_MAT_K[0] == 32) {
      const int KP_K = 32;
      const int TILE_Y = KP_K; //Y direction corresponds to tile of column of the KP factor
      const int TILE_X = 1; //X direction correspond to tile of row 
      const int KP_K_BATCH = 1;
      const int N_COARSE_TB = 1;
      const int TILE_K = KP_K*KP_K;

      dim3 grid = {M/TILE_X/N_COARSE_TB, 1}; //(N/KP_MAT_N[NUM_KP_MATS-i-1])/TILE_Y
      dim3 block = {128,1,1};
      cuda_gemm<int,128,TILE_K,N_COARSE_TB,TILE_Y,TILE_X,TILE_K,KP_K,KP_K,KP_K_BATCH><<<grid, block, 0, stream>>>(M, N, prev_kp, kpMats[NUM_KP_MATS-i-1], kpMatmulResult[i]);
    }
    // CUDACHECK(cudaDeviceSynchronize());
  }
}

bool check(int* ref, int* computed, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (ref[i*N + j] != computed[i* N + j]) {
        printf("Mismatch for %d x %d at (%d, %d): ref = %d, computed = %d\n", M, N, i, j, ref[i*N+j], computed[i*N+j]);
        return false;
      }
    }
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int one(int i, int j) {return 1;}
int zeroOne(int i, int j) {return i % 2;}
int setToI(int i, int j) {return i;}
int randMod(int i, int j) {return rand()%10;}

void setValues(int NUM_KP_MATS, int* kpMats[], int *x, int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[], int (*fnvalue)(int i, int j))
{
  for (int i = 0; i < NUM_KP_MATS; i++) {
    setMatrix(kpMats[i], KP_MAT_K[i], KP_MAT_N[i], fnvalue);
  }

  setMatrix(x, M, K, fnvalue);
}

struct MatrixSizes {
  const int M, N, K;
  const int NUM_KP_MATS;
  const std::vector<int> KP_MAT_N; 
  const std::vector<int> KP_MAT_K;
};

int main(int argc, char* argv[]) 
{
  std::vector<MatrixSizes> matrixSizes = {
                                          // {4,4,4, 2, {2,2},{2,2}},
                                          // {4,4,6, 2, {1,4},{2,3}},
                                          // {4,4,8, 2, {2,2},{2,4}},
                                          // {4,4,8, 2, {2,2},{4,2}},
                                          // {8,8,8, 2, {4,2},{4,2}},
                                          // {8,8,8, 2, {4,2},{2,4}},
                                          // {8,8,8, 3, {2,2,2},{2,2,2}},
                                          // {8,8,32, 3, {2,2,2},{2,4,4}},
                                          // {8,16,32, 3, {4,2,2},{2,4,4}},
                                          // {8,8,16, 3, {2,2,2},{2,4,2}},
                                          // {16,8,8, 3, {2,2,2},{2,2,2}},
                                          // {16,16,16, 2, {4,4},{4,4}},
                                          // {16,16,16, 3, {4,2,2},{4,2,2}},
                                          // {16,16,16, 3, {4,2,2},{2,4,2}},
                                          // {16,16,16, 3, {8,2,1},{2,4,2}},
                                          // {16,16,16, 4, {2,2,2,2},{2,2,2,2}},
                                          // {16,16,64, 4, {2,2,2,2},{2,4,2,4}},
                                          // {256,256,256, 4, {4,4,4,4},{4,4,4,4}},
                                          // {256,256,256, 2, {16,16},{16,16}},
  #ifdef EVAL
                                          // {65536,1024,1024, 2, {32,32},{32,32}},
                                          // {65536,256,256, 2, {16,16},{16,16}},
                                          {65536,512,512, 3, {8,8,8},{8,8,8}},
                                          
                                          // {1024,32*1024,32*1024, 2, {32,32,32},{32,32,32}},
  #else
                                          {512,1024,1024, 2, {32,32},{32,32}},
                                          {512,256,256, 2, {16,16},{16,16}},
                                          {512,512,512, 3, {8,8,8},{8,8,8}},
  #endif

                                          // {1024, 1024, 1024, 2, {32,32},{32,32}}
                                          };

  // int (*fnvalues[4])(int, int) = {&one, &zeroOne, &setToI, &randMod};
  int (*fnvalues[1])(int, int) = {&randMod};

  for (MatrixSizes matrixSize : matrixSizes) {
    int M = matrixSize.M;
    int N = matrixSize.N;
    int K = matrixSize.K;
    
    int NUM_KP_MATS = matrixSize.NUM_KP_MATS;
    int KP_MAT_N[NUM_KP_MATS];
    int KP_MAT_K[NUM_KP_MATS];

    printf("Matmul: %d x %d x %d, Num KP Factors: %d\n", M, N, K, NUM_KP_MATS);
    int n=1,k=1;
    for (int i = 0; i < NUM_KP_MATS; i++) {
      k *= matrixSize.KP_MAT_K[i];
      n *= matrixSize.KP_MAT_N[i];
    }
    if (n != N || k != K) {
      printf("Invalid KP Factors Sizes %d != %d, %d != %d\n", n, N, k, K);
    }

    int *kpout[NUM_KP_MATS];
    int *kpMats[NUM_KP_MATS];
    int* kpMatmulResult[NUM_KP_MATS];

    int *x = new int[M*K];

    int* dX;
    int** dKpOut;
    int** dKpMats;
    int** dKpMatmulResult;
    
    CUDACHECK(cudaMalloc(&dX, M*K * sizeof(int)));
    CUDACHECK(cudaMalloc(&dKpMats, NUM_KP_MATS * sizeof(int*)));
    CUDACHECK(cudaMalloc(&dKpMatmulResult, NUM_KP_MATS * sizeof(int*)));
    CUDACHECK(cudaMalloc(&dKpOut, NUM_KP_MATS * sizeof(int*)));

    int* __dKpOut[NUM_KP_MATS];
    int* __dKpMats[NUM_KP_MATS];
    int* __dKpMatmulResult[NUM_KP_MATS];

    for (int i = 0; i < NUM_KP_MATS; i++) {
      KP_MAT_K[i] = matrixSize.KP_MAT_K[i];
      KP_MAT_N[i] = matrixSize.KP_MAT_N[i];
      kpMats[i] = new int[KP_MAT_K[i] * KP_MAT_N[i]];
      kpout[i] = new int[K*N]; //TODO: larger than needed
      kpMatmulResult[i] = new int[M*std::max(N,K)];

      CUDACHECK(cudaMalloc(&__dKpMats[i], KP_MAT_K[i] * KP_MAT_N[i] * sizeof(int)));
      // CUDACHECK(cudaMalloc(&__dKpOut[i], K * N * sizeof(int)));
      CUDACHECK(cudaMalloc(&__dKpMatmulResult[i], M*std::max(N,K) * sizeof(int)));

      CUDACHECK(cudaMemset(__dKpMatmulResult[i], 0, M*std::max(N,K) * sizeof(int)));
      // CUDACHECK(cudaMemset(__dKpOut[i], 0, K * N * sizeof(int)));
    }

    // CUDACHECK(cudaMemcpy(&dKpOut[0], &__dKpOut[0], NUM_KP_MATS * sizeof(int*), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(&dKpMats[0], &__dKpMats[0], NUM_KP_MATS * sizeof(int*), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(&dKpMatmulResult[0], &__dKpMatmulResult[0], NUM_KP_MATS * sizeof(int*), cudaMemcpyHostToDevice));

    int* result = new int[M*N];

    int* dResult;

    CUDACHECK(cudaMalloc(&dResult, M * N * sizeof(int)));

    for (int fnvalue = 0; fnvalue < sizeof(fnvalues)/sizeof(fnvalues[0]); fnvalue++) {
      setValues(NUM_KP_MATS, kpMats, x, M, N, K, KP_MAT_N, KP_MAT_K, fnvalues[fnvalue]);

      for (int i = 0; i < NUM_KP_MATS; i++) {
        CUDACHECK(cudaMemcpy(__dKpMats[i], kpMats[i], KP_MAT_K[i] * KP_MAT_N[i] * sizeof(int), cudaMemcpyHostToDevice));
      }
    
      CUDACHECK(cudaMemcpy(dX, x, M * K * sizeof(int), cudaMemcpyHostToDevice));
  #ifndef EVAL
      baselineKPThenMatmul(NUM_KP_MATS, result, x, kpout, kpMats, 
                           M, N, K, KP_MAT_N, KP_MAT_K);
  #endif
      // slicedMatmul(NUM_KP_MATS, kpMatmulResult, x, kpMats,
      //              M, N, K, KP_MAT_N, KP_MAT_K);

      for (int i = 0; i < NUM_KP_MATS; i++)
        CUDACHECK(cudaMemset(__dKpMatmulResult[i], 0, M*std::max(N,K) * sizeof(int)));
  #ifdef EVAL  
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      cudaEvent_t start;
      cudaEvent_t end;
      float elapsedTime;
      CUDACHECK(cudaEventCreate(&start));
      CUDACHECK(cudaEventCreate(&end));
      for (int i = 0; i < 10; i++)
        customKronGEMM(NUM_KP_MATS, __dKpMatmulResult, dX, __dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, stream);
      CUDACHECK(cudaStreamSynchronize(stream));
      CUDACHECK(cudaEventRecord(start, stream));
      for (int i = 0; i < 100; i++)
        customKronGEMM(NUM_KP_MATS, __dKpMatmulResult, dX, __dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, stream);
      CUDACHECK(cudaEventRecord(end, stream));
      CUDACHECK(cudaEventSynchronize(end));
      CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, end));
      printf("elapsedtime %f\n", elapsedTime);
      return;
  #else
      for (int i = 0; i < 1; i++)
        customKronGEMM(NUM_KP_MATS, __dKpMatmulResult, dX, __dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, 0);
      CUDACHECK(cudaDeviceSynchronize());
  #endif
      // return;
      int* hKpMatMulResult = new int[M*N];
      // return;
      for (int i = 0; i < NUM_KP_MATS; i++)
        CUDACHECK(cudaMemcpy(kpMatmulResult[i], __dKpMatmulResult[i], M*N*sizeof(int), cudaMemcpyDeviceToHost));
      // if (check(result, kpMatmulResult[NUM_KP_MATS-1], M, N))
      if (check(result, kpMatmulResult[NUM_KP_MATS-1], M,N))
        printf("Results Correct for test %d\n", fnvalue);
      else {
        // printf("\nMatmul:");
        // printMatrix(result, K, N);

        // printf("\nx:");
        // printMatrix(x, M, K);    
        // for (int kpMatId = 0; kpMatId < NUM_KP_MATS; kpMatId++) {
        //   printf("\nKP Mat %d:", kpMatId);
        //   printMatrix(kpMats[kpMatId], KP_MAT_K[kpMatId], KP_MAT_N[kpMatId]);
        // }
        // // printf("\nKP Out:");
        // // printMatrix(kpout[0], 8, 8);
        // for (int id = 0; id < NUM_KP_MATS; id++) {
        //   printf("\nKP result %d:", id);
        //   printMatrix(kpMatmulResult[id], M, N);
        // }
        // printf("\nKP result 2:");
        // printMatrix(kpMatmulResult[2], 16, 16);
        // printf("\nKP result 3:");
        // printMatrix(kpMatmulResult[3], 16, 16);
        // printf("\nKP result 1:");
        // printMatrix(kpMatmulResult[1], M, N);
        // printf("\n");
        return 0;
      }
    }

    //Is there really a need to free anything when you have tons of RAM, am I right?
  }

  return 0;
}