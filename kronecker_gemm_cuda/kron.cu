
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
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define DIVUP(x, y) (((x) + (y) - 1)/((y)))

template<typename T>
void setMatrix(T* mat, int M, int N, int (*fnvalue)(int i, int j)) 
{
  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      mat[i*N + j] = (T)fnvalue(i,j);
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

#define EXTERNAL_KP_K_TILE_ 64

__device__ bool isfirstIdx(dim3 idx) {return idx.x == 0 && idx.y == 0 & idx.z == 0;}

template<typename T,int N_THREADS, int N_COARSE_TB, int TILE_X, int MAX_K, int MAX_KP_N, int MAX_KP_K, int KP_N_TILE_, int K_EQUALS_VAR, int KPK_EQUALS_VAR>
__global__ void __launch_bounds__(N_THREADS) cuda_gemm(int M, int NVar, int KVar, T * A, T * kron_fac, T * C, int kpNVar, int kpKVar, int kp_idx) {
  const int KP_N_TILE = MIN(KP_N_TILE_, MAX_KP_N);
  const int NUM_KP_N_TILES = MAX_KP_N/KP_N_TILE;
  const int INTERNAL_KP_N_TILE = MIN(16, KP_N_TILE);
  const int EXTERNAL_KP_K_TILE = MIN(EXTERNAL_KP_K_TILE_, MAX_KP_K);
  const int INTERNAL_KP_K_TILE = MIN(32, EXTERNAL_KP_K_TILE);

  #ifdef EVAL
    typedef float4 LD_TYPE; 
  #else 
    typedef int4 LD_TYPE; 
  #endif 

  __shared__ __align__(128) T kron_fac_sh[INTERNAL_KP_N_TILE][INTERNAL_KP_K_TILE+1];//TODO: Change padding based on value o1, KP_K and TILE_Y
  const int Ash_COLS = MAX_K/(MAX_KP_K/INTERNAL_KP_K_TILE);
  __shared__ __align__(128) T Ash[TILE_X][Ash_COLS];
  const int C_ELEMS_STORE = N_THREADS * (sizeof(LD_TYPE)/sizeof(T));
  const int Csh_COLS = MAX_K/(MAX_KP_N/KP_N_TILE);
  const int Csh_COLS_SIZE = MIN(Csh_COLS, C_ELEMS_STORE);
  __shared__ __align__(128) T Csh[TILE_X][Csh_COLS];//Allocate Csh for only as many values that are produced

  int wid = threadIdx.x/32;
  int lane = threadIdx.x%32;
  int blockWarps = blockDim.x/32;
  int kpK;
  int kpN;
  int K;
  int N;
 
  if (KPK_EQUALS_VAR) {
    kpK = MAX_KP_K;
    kpN = MAX_KP_N;
  } else {
    kpK = kpKVar;
    kpN = kpNVar;
  }

  if (K_EQUALS_VAR) {
    K = MAX_K;
    N = K;
  } else {
    K = KVar;
    N = NVar;
  }

  const int KPK_SPLIT_SIZE = MIN(16, INTERNAL_KP_K_TILE);
  const int NUM_KPK_SPLITS = MAX(1, INTERNAL_KP_K_TILE/KPK_SPLIT_SIZE);

  int external_tile_kp_n;
  int tile_k;
  int external_tile_kp_k = blockIdx.z;
  
  if (KP_N_TILE == MAX_KP_N && INTERNAL_KP_N_TILE == MAX_KP_N && INTERNAL_KP_K_TILE == MAX_KP_K) {
    for (auto i = threadIdx.x; i < kpN * kpK; i += blockDim.x) {
      kron_fac_sh[i%kpN][i/kpK] = kron_fac[i];
    }

    external_tile_kp_n = 0;
    tile_k = blockIdx.x;
  } else {
    external_tile_kp_n = blockIdx.x%DIVUP(MAX_KP_N, KP_N_TILE);
    tile_k = blockIdx.x/DIVUP(MAX_KP_N, KP_N_TILE);
    // if (K/MAX_K == 32)  {
    //   external_tile_kp_n = blockIdx.x / 32;
    //   tile_k = blockIdx.x % 32;
    // } else if (K/MAX_K == 64) {
    //   external_tile_kp_n = blockIdx.x / 64;
    //   tile_k = blockIdx.x % 64;
    // } else {
    //   //TODO: Optimize by making more special cases
    //   external_tile_kp_n = blockIdx.x/(K/MAX_K);
    //   tile_k = blockIdx.x % (K/MAX_K);
    // }
  }

  const int ldNumElems = (sizeof(LD_TYPE)/sizeof(T));
  
  const int numKpColMult = MIN(MAX_K/kpK*NUM_KPK_SPLITS, N_THREADS); //Threads executing in parallel to multiply one column of KP with MAX_K row elements of A //128, 16, 128
  const int kpMulblockWarps = N_THREADS/numKpColMult; //1 , 8, 1
  const int Creg_SIZE = MAX(1, Csh_COLS/N_THREADS); // 8 , 2, 16
  const int Creg_Rows = (MAX_K/kpK)/numKpColMult; //1024/4/128 = 2, 1, 4
  const int Creg_Cols = INTERNAL_KP_N_TILE/kpMulblockWarps; //4/1 = 4, 2, 4

  register T Creg[Creg_SIZE];

  int kpSplitLane = threadIdx.x%NUM_KPK_SPLITS;
  int kpMullane = (threadIdx.x%numKpColMult)/NUM_KPK_SPLITS;
  int kpMulwid = threadIdx.x/numKpColMult; //0
   //TODO: Names should be different

  for (int start_row = blockIdx.y * TILE_X; start_row < gridDim.y * TILE_X * N_COARSE_TB; start_row += gridDim.y * TILE_X) {
    if (NUM_KPK_SPLITS != 1) {
      for (int a_row = 0; a_row < TILE_X; a_row += 1) {
        for (int i = threadIdx.x; i < Csh_COLS; i += blockDim.x)
          Csh[a_row][i] = 0;
      }
    } else {
      #pragma unroll
      for (int reg = 0; reg < Creg_SIZE; reg++) {
        Creg[reg] = 0;
      }
    }

    for (int internal_tile_kp_k = 0; internal_tile_kp_k < EXTERNAL_KP_K_TILE; internal_tile_kp_k += INTERNAL_KP_K_TILE) {
      for (int a_row = 0; a_row < TILE_X; a_row += 1) {
        for (int a_col = threadIdx.x*ldNumElems; a_col < Ash_COLS; a_col += blockDim.x*ldNumElems) {
          if (INTERNAL_KP_K_TILE == MAX_KP_K) {
            LD_TYPE a = *(LD_TYPE*)&A[(a_row + start_row) * K + (K_EQUALS_VAR ? 0 : tile_k*MAX_K) + a_col];

            *(LD_TYPE*)&Ash[a_row][a_col] = a;
          } else {
            LD_TYPE a = *(LD_TYPE*)&A[(a_row + start_row) * K + (K_EQUALS_VAR ? 0 : tile_k*MAX_K) + \
                                      (a_col/INTERNAL_KP_K_TILE)*kpK + external_tile_kp_k * EXTERNAL_KP_K_TILE + internal_tile_kp_k + a_col % INTERNAL_KP_K_TILE];
            
            *(LD_TYPE*)&Ash[a_row][a_col] = a;
          }
        }      
      }
    
      //TODO: nvcc unrolls this loop, which leads to high register usage
      for (int internal_tile_kp_n = 0; internal_tile_kp_n < KP_N_TILE; internal_tile_kp_n += INTERNAL_KP_N_TILE) {
        if (!(KP_N_TILE == MAX_KP_N && INTERNAL_KP_N_TILE == MAX_KP_N && INTERNAL_KP_K_TILE == MAX_KP_K)) {
          //Create kpK subwarps and each subwarp loads 0 to INTERNAL_KP_N_TILE elements
    
          for (int swid = threadIdx.x/INTERNAL_KP_N_TILE; swid < INTERNAL_KP_K_TILE; swid += blockDim.x/INTERNAL_KP_N_TILE) {
            int col = external_tile_kp_n*KP_N_TILE + internal_tile_kp_n + threadIdx.x%INTERNAL_KP_N_TILE;
            int row = swid;
            kron_fac_sh[threadIdx.x%INTERNAL_KP_N_TILE][row] = kron_fac[(external_tile_kp_k * EXTERNAL_KP_K_TILE + internal_tile_kp_k + row) * kpN + col];
          }
        }

        __syncthreads();

        for (int a_row = 0; a_row < TILE_X; a_row++) {
          for (int a_col_start = 0, c_reg_col_start = 0; a_col_start < MAX_K/kpK; a_col_start += numKpColMult, c_reg_col_start++) {
            const int MAX_AR_SZ = MIN(KPK_SPLIT_SIZE, MAX_KP_K);

            //Load MAX_AR_SZ elements at a time to limit the register usage
            // for (int ar_start = 0; ar_start < MAX_KP_K; ar_start += MAX_AR_SZ) 
            {
              register T Ar[MAX_AR_SZ];
              int kpKlane = kpMullane % MAX_AR_SZ;
              int ar_start = kpSplitLane * MAX_AR_SZ;

              for (int a_col = kpKlane, i = 0; i < MAX_AR_SZ; i++) { //
                if (INTERNAL_KP_K_TILE == MAX_KP_K) {
                  if (i < kpK) {
                    Ar[i] = Ash[a_row][(a_col_start+kpMullane)*kpK + ar_start + (a_col + i) % MAX_AR_SZ];//TODO: Shared memory bank conflicts here with KP_K = 4
                  }
                } else {
                  Ar[i] = Ash[a_row][(a_col_start+kpMullane)*INTERNAL_KP_K_TILE + ar_start + (a_col + i) % MAX_AR_SZ];//TODO: Shared memory bank conflicts here with KP_K = 4
                }
              }
              
              for (int kp_col = kpMulwid, creg_idx = 0; kp_col < min(kpN, INTERNAL_KP_N_TILE); kp_col += kpMulblockWarps, creg_idx++) {
                T c = 0;

                register T kron_fac_r;

                kron_fac_r = kron_fac_sh[kp_col][lane % INTERNAL_KP_K_TILE];
                
                #pragma unroll
                for (int a_col = 0; a_col < MIN(MAX_KP_K, MAX_AR_SZ); a_col++) {
                  if (a_col < kpK) {
                    T a = Ar[a_col]; //Ash[a_row][a_col_start/KP_K][a_col]; //Ar[a_col];
                    int kp_row;
                    if (KPK_EQUALS_VAR) {
                      kp_row = ar_start + (a_col + kpKlane)%min(kpK, KPK_SPLIT_SIZE); //kpMullane/(warpSize/kpK)
                    } else {kp_row = (a_col+kpKlane) < kpK ? (a_col+kpKlane) : (a_col+kpKlane) - kpK;} //TODO:
                    T kp;
                    if (INTERNAL_KP_K_TILE <= 32 && kpK <= 64) {
                      // kp = kron_fac_sh[kp_col][ar_start+(a_col+kpKlane)%min(kpK, KPK_SPLIT_SIZE)];
                      kp = __shfl_sync(0xffffffff, kron_fac_r, kp_row, INTERNAL_KP_K_TILE);
                      // if (kp_col == 0 && ar_start == 16 && kpK == 128 && kp != kp1 && isfirstIdx(blockIdx))
                      //   printf("kp_col %d kp_row %d %d, %d %d, %d %d %d\n", kp_col, kp_row, ar_start + (a_col+kpKlane) % min(MAX_AR_SZ, kpK), kp, kp1, ar_start, a_col, kpKlane);
                    } else {
                      //FIXME: For 1x16384 with 128x128 Kronecker factors, the results are incorrect for __shfl_sync because numkpcolmult != 32
                      // kp_row = ar_start + kpKlane + (a_col+kpKlane < min(MAX_AR_SZ, kpK) ? a_col : a_col - min(MAX_AR_SZ, kpK));
                      kp_row = ar_start + (a_col+kpKlane) % min(MAX_AR_SZ, kpK);
                      kp = kron_fac_sh[kp_col][kp_row];
                      // if (a_row == 0 && kp_col == 0 && kpMullane == 0 && isfirstIdx(blockIdx))
                      //   printf("kpSplitLane %d kp_row %d kp %d internal_tile_kp_k %d\n", kpSplitLane, kp_row, kp, internal_tile_kp_k);
                    } 

                    c += a * kp;
                  }
                }

                // if (a_row == 0 && kp_col == 0 && kpMullane == 0 && isfirstIdx(blockIdx))
                //   printf("kpSplitLane %d c %d internal_tile_kp_k %d\n", kpSplitLane, c, internal_tile_kp_k);
                int csh_col = (internal_tile_kp_n + kp_col)*(MAX_K/kpK) + a_col_start +kpMullane;
                if (NUM_KPK_SPLITS == 1) {
                  Creg[c_reg_col_start*Creg_Cols + creg_idx] += c;
                  int c_idx = start_row * N + c_reg_col_start*N_THREADS + creg_idx * (kpMulblockWarps * numKpColMult * (MAX_K/kpK)/numKpColMult) + threadIdx.x;
                  // C[c_idx] = c;
                // Csh[a_row][csh_col] += c;
                }
                else {
                  for (int offset = NUM_KPK_SPLITS/2; offset > 0; offset /= 2) 
                    c += __shfl_down_sync(0xffffffff, c, offset, NUM_KPK_SPLITS);
                  if (kpSplitLane == 0) {
                    Csh[a_row][csh_col] += c;
                  }
                }
              }
            }
          }
        }
      }
    }

    __syncthreads();
    
    if (NUM_KPK_SPLITS == 1) {
      for (int reg = 0; reg < Creg_SIZE; reg++) {
        int a_row = 0;
        int c_row = (a_row + start_row);
        int c_idx;
        if (K_EQUALS_VAR)
          c_idx = start_row * N + (reg/Creg_Cols)*N_THREADS + (reg%Creg_Cols) * (kpMulblockWarps * numKpColMult * (MAX_K/kpK)/numKpColMult) + threadIdx.x;
        else
          c_idx = start_row * N + tile_k * (MAX_K/kpK) + (reg/Creg_Cols)*N_THREADS + (reg%Creg_Cols) * (N_THREADS * (MAX_K/kpK)/numKpColMult) * (K/MAX_K) + threadIdx.x;
         
        C[c_idx] = Creg[reg];
      }
    } else {
      for (int a_row = 0; a_row < TILE_X; a_row++) {
        if (EXTERNAL_KP_K_TILE != MAX_KP_K) {
          //Atomic Store when there is an external KP_K tile
          for (int c_col = threadIdx.x; c_col < Csh_COLS; c_col += blockDim.x) {
            int c_row = (a_row + start_row);
            int c_idx;
            if (K_EQUALS_VAR)
              c_idx = c_row * N + external_tile_kp_n*Csh_COLS + c_col;
            else
              c_idx = c_row * N + external_tile_kp_n*(K/(MAX_KP_N/KP_N_TILE)) + tile_k * (MAX_K/kpK) + (c_col/(MAX_K/kpK)) * (K/kpK) + c_col%(MAX_K/kpK);
            
            atomicAdd(&C[c_idx], Csh[a_row][c_col]);
            // C[c_idx] = Csh[a_row][c_col];

            // if (kp_idx == 0 && c_idx >= 2048 && c_idx < 2048+64) {
            //   printf("Csh[a_row][%d] %d tile_k %d C[c_idx] %d\n", c_col, Csh[a_row][c_col], tile_k, C[c_idx]);
            // }
          }
        } else {
          //Normal Store
          for (int c_col = threadIdx.x*ldNumElems; c_col < Csh_COLS; c_col += blockDim.x*ldNumElems) {
            int c_row = (a_row + start_row);
            int c_idx;
            if (K_EQUALS_VAR)
              c_idx = c_row * N + external_tile_kp_n*Csh_COLS + c_col;
            else
              c_idx = c_row * N + external_tile_kp_n*(K/(MAX_KP_N/KP_N_TILE)) + tile_k * (MAX_K/kpK) + (c_col/(MAX_K/kpK)) * (K/kpK) + c_col%(MAX_K/kpK);
            
            *(LD_TYPE*)&C[c_idx] = *(LD_TYPE*)&Csh[a_row][c_col];
          }
        }
      }
    }
  }
}

// #define KP_EQUALS_VAR_KERNELS (N_COARSE_TB, MAX_K, KP_N_K, K_EQUALS_VAR) \
//   (void*)cuda_gemm<int,128,N_COARSE_TB,1,MAX_K,KP_N_K,KP_N_K,0>,\
//   (void*)cuda_gemm<int,128,N_COARSE_TB,1,MAX_K,KP_N_K,KP_N_K,1>,

#define N_THREADS 128
#define KP_N_TILE 64

#ifdef EVAL
    typedef float DATA_TYPE;
  #else
    typedef int DATA_TYPE;
  #endif

#define TILE_X 1

#define K_EQUALS_VAR_KERNELS(N_COARSE_TB, MAX_K, KP_N_K, K_EQUALS_VAR) \
  (void*)cuda_gemm<DATA_TYPE,N_THREADS,N_COARSE_TB,TILE_X,MAX_K,KP_N_K,KP_N_K,KP_N_TILE,K_EQUALS_VAR,0>,\
  (void*)cuda_gemm<DATA_TYPE,N_THREADS,N_COARSE_TB,TILE_X,MAX_K,KP_N_K,KP_N_K,KP_N_TILE,K_EQUALS_VAR,1>,

#define KP_N_K_KERNELS(N_COARSE_TB, MAX_K, KP_N_K) \
  K_EQUALS_VAR_KERNELS(N_COARSE_TB, MAX_K, KP_N_K, 0) \
  K_EQUALS_VAR_KERNELS(N_COARSE_TB, MAX_K, KP_N_K, 1)

#define MAX_K_KERNELS(N_COARSE_TB, MAX_K) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 2) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 4) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 8) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 16) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 32) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 64) \
  KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 128) 

#define COARSE_TB_KERNELS(N_COARSE_TB) \
  MAX_K_KERNELS(N_COARSE_TB, 16) \
  MAX_K_KERNELS(N_COARSE_TB, 32) \
  MAX_K_KERNELS(N_COARSE_TB, 64) \
  MAX_K_KERNELS(N_COARSE_TB, 128) \
  MAX_K_KERNELS(N_COARSE_TB, 256) \
  MAX_K_KERNELS(N_COARSE_TB, 512) \
  MAX_K_KERNELS(N_COARSE_TB, 1024) \
  MAX_K_KERNELS(N_COARSE_TB, 2048) \
  MAX_K_KERNELS(N_COARSE_TB, 4096) \

#define MAX_K 2048
#define NUM_MAX_K_KERNELS 9
#define NUM_KP_N_K_KERNELS 7
#define NUM_COARSE_TB_KERNELS 3
#define NUM_K_EQUALS_VAR 2
#define NUM_KPK_EQUALS_VAR 2

static void* cudaGemmSpecialized[NUM_COARSE_TB_KERNELS][NUM_MAX_K_KERNELS][NUM_KP_N_K_KERNELS][NUM_K_EQUALS_VAR][NUM_KPK_EQUALS_VAR] = {
  // KP_N_K_KERNELS(8, 1024, 32)
    COARSE_TB_KERNELS(1)
    COARSE_TB_KERNELS(2)
    COARSE_TB_KERNELS(4)
  };

static_assert(sizeof(cudaGemmSpecialized)/sizeof(void*) == NUM_COARSE_TB_KERNELS * NUM_KP_N_K_KERNELS * NUM_MAX_K_KERNELS*NUM_K_EQUALS_VAR*NUM_KPK_EQUALS_VAR);

int log2(int n){return 31 - __builtin_clz(n);}

template<typename T>
T* customKronGEMM(const int NUM_KP_MATS, T* kpMatmulResult[], T* x, T* kpMats[],
                    int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[], cudaStream_t stream)
{
  typedef int (*cuda_gemm_ty)(int, int, int, T*, T*, T*, int kpNVar, int kpKVar);

  //Row Major Layout of all matrics
  T* resultMat = kpMatmulResult[0];
  T* prevResult = x;
  for (int i = 0; i < NUM_KP_MATS; i++) {

    const int KP_K_BATCH = 1;
    int N_COARSE_TB = (M > 100) ? 2 : 1;

    // int idx = (N_COARSE_TB/8)*NUM_MAX_K_KERNELS + (log2(K)-log2(16))*NUM_KP_N_K_KERNELS + (log2(KP_MAT_K[0])-log2(2));
    // printf("idx %d log2(K) %d log2(16) %d\n", idx, log2(K), log2(16));
    // assert(idx < sizeof(cudaGemmSpecialized)/sizeof(void*));
    
    int min_k = min(K, MAX_K);
    int k_equals_var = (min_k == K) ? 1 : 0;
    // if (min_k/KP_MAT_K[0] >= 256) {
    //   //K dimension is very high. Divide it in different threadblocks to have better parallelism
    //   min_k = min_k/KP_MAT_K[0];
    //   k_equals_var = 0;
    // }
    cuda_gemm_ty cuda_gemm_func = (cuda_gemm_ty)cudaGemmSpecialized[N_COARSE_TB/2][log2(min_k)-log2(16)][log2(KP_MAT_K[0])-log2(2)][k_equals_var][1];
    dim3 grid = {(K/min_k) * DIVUP(KP_MAT_N[0], KP_N_TILE), DIVUP((M/TILE_X), N_COARSE_TB), DIVUP(KP_MAT_K[0], EXTERNAL_KP_K_TILE_)}; 
    dim3 block = {N_THREADS,1,1};

    void *args[] = {&M, &N, &K, &prevResult, (void*)&kpMats[NUM_KP_MATS-i-1], (void*)&resultMat, (void*)&KP_MAT_N[NUM_KP_MATS-i-1], (void*)&KP_MAT_K[NUM_KP_MATS-i-1], &i};

    CUDACHECK(cudaLaunchKernel((const void*)cuda_gemm_func, grid, block, &args[0], 0, stream));

    if (i < NUM_KP_MATS - 1) {
      prevResult = resultMat;
      if (resultMat == kpMatmulResult[0]) {        
        resultMat = kpMatmulResult[1];
      } else if (resultMat == kpMatmulResult[1]) {
        resultMat = kpMatmulResult[0];
      }
    }
    
    CUDACHECK(cudaDeviceSynchronize());
  }

  return resultMat;
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
int randMod(int i, int j) {return rand()%5 + 1;}

template<typename T>
void setValues(int NUM_KP_MATS, T* kpMats[], T *x, int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[], int (*fnvalue)(int i, int j))
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
                                          // {10,1024,1024, 2, {32,32},{32,32}},                                        
                                          {10,1024,1024, 10, {2,2,2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2,2,2}},
                                          {10,256,256, 2, {16,16},{16,16}},
                                          // {10,256,256, 2, {16,16},{16,16}},
                                          {10,512,512, 3, {8,8,8},{8,8,8}},
                                          {10,256,256, 4, {4,4,4,4},{4,4,4,4}},
                                          {10,1024,1024, 5, {4,4,4,4,4},{4,4,4,4,4}},
                                          {4,4096,4096, 6, {4,4,4,4,4,4},{4,4,4,4,4,4}},
                                          // {1, 4096, 4096, 2, {64,64},{64,64}},
                                          // {1, 128*128, 128*128, 2, {128,128},{128,128}}
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

    DATA_TYPE *kpout[NUM_KP_MATS];
    DATA_TYPE *kpMats[NUM_KP_MATS];
    DATA_TYPE* kpMatmulResult[NUM_KP_MATS];

    DATA_TYPE *x = new DATA_TYPE[M*K];

    DATA_TYPE* dX;
    DATA_TYPE** dKpOut;
    DATA_TYPE** dKpMats;
    DATA_TYPE** dKpMatmulResult;
    
    CUDACHECK(cudaMalloc(&dX, M*K * sizeof(DATA_TYPE)));
    
    DATA_TYPE* __dKpOut[NUM_KP_MATS];
    DATA_TYPE* __dKpMats[NUM_KP_MATS];
    DATA_TYPE* __dKpMatmulResult[2];

    for (int i = 0; i < NUM_KP_MATS; i++) {
      KP_MAT_K[i] = matrixSize.KP_MAT_K[i];
      KP_MAT_N[i] = matrixSize.KP_MAT_N[i];
      kpMats[i] = new DATA_TYPE[KP_MAT_K[i] * KP_MAT_N[i]];
      kpout[i] = new DATA_TYPE[K*N]; //TODO: larger than needed
      kpMatmulResult[i] = new DATA_TYPE[M*std::max(N,K)];

      CUDACHECK(cudaMalloc(&__dKpMats[i], KP_MAT_K[i] * KP_MAT_N[i] * sizeof(DATA_TYPE)));
      // CUDACHECK(cudaMalloc(&__dKpOut[i], K * N * sizeof(int)));
      

      // CUDACHECK(cudaMemset(__dKpOut[i], 0, K * N * sizeof(int)));
    }

    // CUDACHECK(cudaMemcpy(&dKpOut[0], &__dKpOut[0], NUM_KP_MATS * sizeof(int*), cudaMemcpyHostToDevice));
    // CUDACHECK(cudaMemcpy(&dKpMats[0], &__dKpMats[0], NUM_KP_MATS * sizeof(DATA_TYPE*), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMalloc(&__dKpMatmulResult[0], M*std::max(N,K) * sizeof(DATA_TYPE)));
    CUDACHECK(cudaMalloc(&__dKpMatmulResult[1], M*std::max(N,K) * sizeof(DATA_TYPE)));
    CUDACHECK(cudaMemset(__dKpMatmulResult[0], 0, M*std::max(N,K) * sizeof(DATA_TYPE)));
    CUDACHECK(cudaMemset(__dKpMatmulResult[1], 0, M*std::max(N,K) * sizeof(DATA_TYPE)));

    DATA_TYPE* result = new DATA_TYPE[M*N];

    DATA_TYPE* dResult;

    CUDACHECK(cudaMalloc(&dResult, M * N * sizeof(DATA_TYPE)));

    for (int fnvalue = 0; fnvalue < sizeof(fnvalues)/sizeof(fnvalues[0]); fnvalue++) {
      setValues(NUM_KP_MATS, kpMats, x, M, N, K, KP_MAT_N, KP_MAT_K, fnvalues[fnvalue]);

      for (int i = 0; i < NUM_KP_MATS; i++) {
        CUDACHECK(cudaMemcpy(__dKpMats[i], kpMats[i], KP_MAT_K[i] * KP_MAT_N[i] * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
      }
    
      CUDACHECK(cudaMemcpy(dX, x, M * K * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
  #ifndef EVAL
      baselineKPThenMatmul(NUM_KP_MATS, result, x, kpout, kpMats, 
                           M, N, K, KP_MAT_N, KP_MAT_K);
  #endif
      // slicedMatmul(NUM_KP_MATS, kpMatmulResult, x, kpMats,
      //              M, N, K, KP_MAT_N, KP_MAT_K);

      for (int i = 0; i < 2; i++)
        CUDACHECK(cudaMemset(__dKpMatmulResult[i], 0, M*std::max(N,K) * sizeof(DATA_TYPE)));
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
      }

      CUDACHECK(cudaFree(__dKpMatmulResult[0]));
      CUDACHECK(cudaFree(__dKpMatmulResult[1]));
      CUDACHECK(cudaFree(dX));
      CUDACHECK(cudaFree(dResult));
      continue;
  #else
      
      DATA_TYPE* dResult = customKronGEMM(NUM_KP_MATS, __dKpMatmulResult, dX, __dKpMats, M, N, K, KP_MAT_N, KP_MAT_K, 0);
      CUDACHECK(cudaDeviceSynchronize());
  #endif
      // return;
      #ifndef EVAL 
      DATA_TYPE* hKpMatMulResult = new DATA_TYPE[M*N];
      // return;
      // for (int i = 0; i < NUM_KP_MATS; i++)
      //   CUDACHECK(cudaMemcpy(kpMatmulResult[i], __dKpMatmulResult[i], M*N*sizeof(int), cudaMemcpyDeviceToHost));
      CUDACHECK(cudaMemcpy(kpMatmulResult[NUM_KP_MATS-1], dResult, M*N*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
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
      #endif
    }

    //Is there really a need to free anything when you have tons of RAM, am I right?
  }

  return 0;
}