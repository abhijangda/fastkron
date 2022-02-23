
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

#define MIN(x,y) (((x) < (y)) ? (x) : (y))

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


template<typename T,int N_THREADS, int N_COARSE_TB, int TILE_X, int MAX_K, int MAX_KP_N, int MAX_KP_K, int CONSTS_AND_VARS_SAME>
__global__ 
void __launch_bounds__(N_THREADS) cuda_gemm(int M, int NVar, int KVar, T * A, T * kron_fac, T * C, int kpNVar, int kpKVar) {
  __shared__ __align__(128) int kron_fac_sh[MAX_KP_N][MAX_KP_K+1];//TODO: Change padding based on value o1, KP_K and TILE_Y
  __shared__ __align__(128) int Ash[TILE_X][MAX_K]; //TODO: Add Padding of 4 so that 128-bit loads can be loaded without misaligned address but ideally padding of 1 will be best for shared memory loads of As at line 293
  __shared__ __align__(128) int Csh[TILE_X][MAX_K];

  int wid = threadIdx.x/warpSize;
  int lane = threadIdx.x%warpSize;
  int blockWarps = blockDim.x/warpSize;
  int kpK;
  int kpN;
  int K;
  int N;
 
 
  // return;

  if (CONSTS_AND_VARS_SAME) {
    kpK = MAX_KP_K;
    kpN = MAX_KP_N;
    K = MAX_K;
    N = K;
  } else {
    kpK = kpKVar;
    kpN = kpNVar;
    K = KVar;
    N = NVar;
  }

  //  if (threadIdx.x == 0&& blockIdx.x ==0)printf("%d %d %d %d %d %d\n", KVar, kpK, MAX_K, MAX_KP_K, N_COARSE_TB, CONSTS_AND_VARS_SAME);

  for (auto i = threadIdx.x; i < kpN * kpK; i += blockDim.x) {
    kron_fac_sh[i%kpN][i/kpK] = kron_fac[i];
  }

  typedef int LD_TYPE;
  const int ldNumElems = (sizeof(LD_TYPE)/sizeof(int));
  
  int tile_k = 0;
  const int numKpColMult = min(MAX_K/kpK, N_THREADS);

  int kpMullane = threadIdx.x%numKpColMult;
  int kpMulwid = threadIdx.x/numKpColMult;
  int kpMulblockWarps = blockDim.x/numKpColMult; //TODO: Names should be different

  for (int start_row = blockIdx.y * TILE_X; start_row < gridDim.y * TILE_X * N_COARSE_TB; start_row += gridDim.y * TILE_X) {
    for (int a_row = 0; a_row < TILE_X; a_row += 1) {
      for (int a_col = threadIdx.x*ldNumElems; a_col < MAX_K; a_col += blockDim.x*ldNumElems) {
        LD_TYPE a = *(LD_TYPE*)&A[(a_row + start_row) * K + ((CONSTS_AND_VARS_SAME) ? 0 : blockIdx.x*MAX_K) + a_col];
        
        *(LD_TYPE*)&Ash[a_row][a_col] = a;        
      }
    }

    __syncthreads();

    for (int a_row = 0; a_row < TILE_X; a_row++) {
      for (int i = threadIdx.x; i < MAX_K; i += blockDim.x)
        Csh[a_row][i] = 0;

      for (int a_col_start = 0; a_col_start < MAX_K/kpK; a_col_start += numKpColMult) {
        const int MAX_AR_SZ = 32;

        //Load MAX_AR_SZ elements at a time to limit the register usage
        for (int ar_start = 0; ar_start < MAX_KP_K; ar_start += MAX_AR_SZ) {
          register int Ar[MIN(MAX_AR_SZ, MAX_KP_K)];
          int kpKlane = lane % min(MAX_AR_SZ, kpK);

          for (int a_col = kpKlane, i = 0; i < MIN(MAX_AR_SZ, MAX_KP_K); i++) { //
            if (i < kpK - ar_start) {
              Ar[i] = Ash[a_row][(a_col_start+kpMullane)*kpK + ar_start + (a_col + i < min(MAX_AR_SZ, kpK) ? a_col: a_col - min(MAX_AR_SZ, kpK)) + i];//TODO: Shared memory bank conflicts here with KP_K = 4
            }
          }

          for (int kp_col = kpMulwid; kp_col < kpN; kp_col += kpMulblockWarps) {
            register int kron_fac_r;
            
            kron_fac_r = kron_fac_sh[kp_col][ar_start+kpKlane];

            //for (int a_col_start = lane * KP_K; a_col_start < TILE_K; a_col_start += KP_K*KP_K) {
            {
              int c = 0;

              #pragma unroll
              for (int a_col = 0; a_col < MIN(MAX_KP_K, MAX_AR_SZ); a_col++) {
                if (a_col < kpK - ar_start) {
                  int a = Ar[a_col]; //Ash[a_row][a_col_start/KP_K][a_col]; //Ar[a_col];
                  int kp_row;
                  if (CONSTS_AND_VARS_SAME) {
                    kp_row = (a_col + kpKlane)%kpK; //kpMullane/(warpSize/kpK)
                  } else {kp_row = (a_col+kpKlane) < kpK ? (a_col+kpKlane) : (a_col+kpKlane) - kpK;}
                  int kp;
                  if (kpK <= 32) {
                    kp = __shfl_sync(0xffffffff, kron_fac_r, kp_row, kpK);
                  } else {
                    kp_row = ar_start + kpKlane + (a_col+kpKlane < min(MAX_AR_SZ, kpK) ? a_col : a_col - min(MAX_AR_SZ, kpK));
                    kp = kron_fac_sh[kp_col][kp_row];
                  } 

                  c += a * kp;
                }
              }

              // if (a_row == 0 && (kp_col)*(MAX_K/kpK)+a_col_start+kpMullane && blockIdx.x == 0)
              //   printf("c %d\n", c);
              Csh[a_row][(kp_col)*(MAX_K/kpK)+a_col_start+kpMullane] += c;
            }
          }
        }
      }
    }

    __syncthreads();

    for (int a_row = 0; a_row < TILE_X; a_row++) {
      for (int c_col = threadIdx.x*ldNumElems; c_col < MAX_K; c_col += blockDim.x*ldNumElems) {
        int c_row = (a_row + start_row);
        int c_idx;
        if (CONSTS_AND_VARS_SAME)
          c_idx = c_row * N + c_col;
        else
          c_idx = c_row * N + blockIdx.x * (MAX_K/kpK) + (c_col/(MAX_K/kpK)) * (K/kpK) + c_col%(MAX_K/kpK);
        // if (c_idx == 0) printf("%d\n", Csh[a_row][c_col]);
        *(LD_TYPE*)&C[c_idx] = *(LD_TYPE*)&Csh[a_row][c_col];
      }
    }
  }
}

#define KERNEL_CALL dim3 grid = {M/TILE_X/N_COARSE_TB, 1}; dim3 block = {128,1,1}; cuda_gemm<int,128,N_COARSE_TB,TILE_X,MAX_K,KP_N,KP_K,CONSTS_AND_VARS_SAME><<<grid, block, 0, stream>>>(M, N, K, prev_kp, kpMats[NUM_KP_MATS-i-1], kpMatmulResult[i], KP_MAT_N[NUM_KP_MATS-i-1], KP_MAT_K[NUM_KP_MATS-i-1]);

#define KP_N_K_KERNELS(N_COARSE_TB, MAX_K, KP_N_K) \
  (void*)cuda_gemm<int,128,N_COARSE_TB,1,MAX_K,KP_N_K,KP_N_K,0>,\
  (void*)cuda_gemm<int,128,N_COARSE_TB,1,MAX_K,KP_N_K,KP_N_K,1>,

#define MAX_K_KERNELS(N_COARSE_TB, MAX_K) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 2) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 4) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 8) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 16) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 32) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 64) 

#define COARSE_TB_KERNELS(N_COARSE_TB) \
  MAX_K_KERNELS(N_COARSE_TB, 16) \
  MAX_K_KERNELS(N_COARSE_TB, 32) \
  MAX_K_KERNELS(N_COARSE_TB, 64) \
  MAX_K_KERNELS(N_COARSE_TB, 128) \
  MAX_K_KERNELS(N_COARSE_TB, 256) \
  MAX_K_KERNELS(N_COARSE_TB, 512) \
  MAX_K_KERNELS(N_COARSE_TB, 1024)

#define NUM_MAX_K_KERNELS 7
#define NUM_KP_N_K_KERNELS 6
#define NUM_COARSE_TB_KERNELS 3
#define NUM_CONSTS_AND_VARS_SAME 2

static void* cudaGemmSpecialized[NUM_COARSE_TB_KERNELS][NUM_MAX_K_KERNELS][NUM_KP_N_K_KERNELS][NUM_CONSTS_AND_VARS_SAME] = {
  // KP_N_K_KERNELS(8, 1024, 32)
    COARSE_TB_KERNELS(1)
    COARSE_TB_KERNELS(8)
    COARSE_TB_KERNELS(16)
  };

typedef int (*cuda_gemm_ty)(int, int, int, int*, int*, int*, int kpNVar, int kpKVar);


static_assert(sizeof(cudaGemmSpecialized)/sizeof(void*) == NUM_COARSE_TB_KERNELS * NUM_KP_N_K_KERNELS * NUM_MAX_K_KERNELS*NUM_CONSTS_AND_VARS_SAME);

int log2(int n){return 31 - __builtin_clz(n);}

void customKronGEMM(const int NUM_KP_MATS, int* kpMatmulResult[], int* x, int* kpMats[],
                     int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[], cudaStream_t stream)
{
  //Row Major Layout of all matrics
  for (int i = 0; i < NUM_KP_MATS; i++) {
    int* prev_kp = (i==0) ? x : kpMatmulResult[i-1];

    const int TILE_X = 1; //X direction correspond to tile of row 
    const int KP_K_BATCH = 1;
    int N_COARSE_TB = (M > 100) ? 8 : 1;

    // int idx = (N_COARSE_TB/8)*NUM_MAX_K_KERNELS + (log2(K)-log2(16))*NUM_KP_N_K_KERNELS + (log2(KP_MAT_K[0])-log2(2));
    // printf("idx %d log2(K) %d log2(16) %d\n", idx, log2(K), log2(16));
    // assert(idx < sizeof(cudaGemmSpecialized)/sizeof(void*));
    
    int min_k = min(K, 1024);
    int consts_equals_vars = K <= 1024 ? 1 : 0;
    if (min_k/KP_MAT_K[0] >= 256) {
      //K dimension is very high. Divide it in different threadblocks to have better parallelism
      min_k = min_k/KP_MAT_K[0];
      consts_equals_vars = 0;
    }
    cuda_gemm_ty cuda_gemm_func = (cuda_gemm_ty)cudaGemmSpecialized[N_COARSE_TB/8][log2(min_k)-log2(16)][log2(KP_MAT_K[0])-log2(2)][consts_equals_vars];
    dim3 grid = {(K/min_k), (M/TILE_X/N_COARSE_TB)}; 
    dim3 block = {128,1,1};

    void *args[] = {&M, &N, &K, &prev_kp, (void*)&kpMats[NUM_KP_MATS-i-1], (void*)&kpMatmulResult[i], (void*)&KP_MAT_N[NUM_KP_MATS-i-1], (void*)&KP_MAT_K[NUM_KP_MATS-i-1]};

    CUDACHECK(cudaLaunchKernel((const void*)cuda_gemm_func, grid, block, &args[0], 0, stream));
    
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
int randMod(int i, int j) {return rand()%5;}

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

  #ifdef EVAL
  if (argc < 4) {printf("invalid command args\n"); return 0;}
  int npoints = atoi(argv[1]);
  int d = atoi(argv[2]);
  int twoPowerL = atoi(argv[3]);
  #endif

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
                                          // {65536,512,512, 3, {8,8,8},{8,8,8}},
                                          // {100,1024,1024, 2, {32,32},{32,32}},
                                          // {10,1024,1024, 2, {32,32},{32,32}},
                                          // {1,1024,1024, 2, {32,32},{32,32}},
                                          // {100,256,256, 4, {4,4,4,4},{4,4,4,4}},
                                          // {10,256,256, 4, {4,4,4,4},{4,4,4,4}},
                                          // {1,256,256, 4, {4,4,4,4},{4,4,4,4}},
                                          // {100,1024,1024, 5, {4,4,4,4,4},{4,4,4,4,4}},
                                          // {10,1024,1024, 5, {4,4,4,4,4},{4,4,4,4,4}},
                                          // {1,1024,1024, 5, {4,4,4,4,4},{4,4,4,4,4}},
                                          
                                          // {100,4096,4096, 6, {4,4,4,4,4,4},{4,4,4,4,4,4}},
                                          // {10,4096,4096, 6, {4,4,4,4,4,4},{4,4,4,4,4,4}},
                                          // {1,4096,4096, 6, {4,4,4,4,4,4},{4,4,4,4,4,4}},

                                          // {100,1024,1024, 3, {16,16,4},{16,16,4}},
                                          // {100,256,256, 2, {16,16},{16,16}},
                                          // {10,1024,1024, 5, {4,4,4,4,4},{4,4,4,4,4}},
                                          // {1,1024,1024, 5, {4,4,4,4,4},{4,4,4,4,4}},

                                          // {100,512*8*8,512*8*8, 5, {8,8,8,8,8},{8,8,8,8,8}},

                                          // {100,1024,1024, 10, {2,2,2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2,2,2}},
                                          // {10,1024,1024, 10, {2,2,2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2,2,2}},
                                          // {1,1024,1024, 10, {2,2,2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2,2,2}},
                                          // {1024,32*1024,32*1024, 2, {32,32,32},{32,32,32}},
  #else
                                          {10,1024,1024, 10, {2,2,2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2,2,2}},
                                          {10,1024,1024, 2, {32,32},{32,32}},
                                          {10,256,256, 2, {16,16},{16,16}},
                                          {10,512,512, 3, {8,8,8},{8,8,8}},
                                          {10,256,256, 4, {4,4,4,4},{4,4,4,4}},
                                          {10,1024,1024, 5, {4,4,4,4,4},{4,4,4,4,4}},
                                          {4,4096,4096, 6, {4,4,4,4,4,4},{4,4,4,4,4,4}},
                                          {1, 4096, 4096, 2, {64,64},{64,64}}
  #endif

                                          // {1024, 1024, 1024, 2, {32,32},{32,32}}
                                          };

  // int (*fnvalues[4])(int, int) = {&one, &zeroOne, &setToI, &randMod};
  int (*fnvalues[1])(int, int) = {&randMod};
  
  #ifdef EVAL
  int Msz = 1;
  for (int i = 0; i < d; i++) {
    Msz *= twoPowerL;
  }
  MatrixSizes matrixSize {
    npoints, Msz, Msz, d, std::vector<int>(d, twoPowerL), std::vector<int>(d, twoPowerL)
  };
  matrixSizes.push_back(matrixSize);
  #endif 

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
      float elapsedTime = 0;
      CUDACHECK(cudaEventCreate(&start));
      CUDACHECK(cudaEventCreate(&end));
      for (int i = 0; i < 10; i++)
        customKronGEMM(NUM_KP_MATS, __dKpMatmulResult, dX, __dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, stream);
      CUDACHECK(cudaStreamSynchronize(stream));
      CUDACHECK(cudaEventRecord(start, stream));
      for (int i = 0; i < 1000; i++)
        customKronGEMM(NUM_KP_MATS, __dKpMatmulResult, dX, __dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, stream);
      CUDACHECK(cudaEventRecord(end, stream));
      CUDACHECK(cudaEventSynchronize(end));
      CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, end));
      printf("elapsedtime %f\n", elapsedTime/1000);

      for (int i = 0; i < NUM_KP_MATS; i++) {
        CUDACHECK(cudaFree(__dKpMats[i]));
        CUDACHECK(cudaFree(__dKpMatmulResult[i]));
      }

      CUDACHECK(cudaFree(dX));
      CUDACHECK(cudaFree(dResult));
      continue;
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