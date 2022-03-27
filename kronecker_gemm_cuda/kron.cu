
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

#define EXTERNAL_KP_K_TILE_ 128

#define C_IN_REG

// #define C_IN_SHMEM
template<uint MAX_KP_N, uint KP_N_TILE> __device__ uint get_tile_k() {return blockIdx.x/DIVUP(MAX_KP_N, KP_N_TILE);}
template<uint MAX_KP_N, uint KP_N_TILE> __device__ uint get_external_tile_kp_n() {return blockIdx.x%DIVUP(MAX_KP_N, KP_N_TILE);}

__device__ bool isfirstIdx(dim3 idx) {return idx.x == 0 && idx.y == 0 & idx.z == 0;}

template<typename T, uint N_THREADS, uint N_COARSE_TB, uint TILE_X, uint MAX_K, uint MAX_KP_N, uint MAX_KP_K, uint KP_N_TILE_, uint K_EQUALS_VAR, uint KPK_EQUALS_VAR>
__global__ void __launch_bounds__(N_THREADS) cuda_gemm(uint M, uint NVar, uint KVar, const T * __restrict__ A, const T * __restrict__ kron_fac, T * __restrict__ C, uint kpNVar, uint kpKVar, uint kp_idx) {
  const uint KP_N_TILE = MIN(KP_N_TILE_, MAX_KP_N);
  const uint NUM_KP_N_TILES = MAX_KP_N/KP_N_TILE;
  const uint INTERNAL_KP_N_TILE = MIN(128, KP_N_TILE);
  const uint EXTERNAL_KP_K_TILE = MIN(EXTERNAL_KP_K_TILE_, MAX_KP_K);
  const uint INTERNAL_KP_K_TILE = MIN(32, EXTERNAL_KP_K_TILE);

  #ifdef EVAL
    typedef float4 LD_TYPE; 
  #else 
    typedef int4 LD_TYPE; 
  #endif 

  __shared__ T kron_fac_sh[INTERNAL_KP_N_TILE][INTERNAL_KP_K_TILE+1];
  const uint Ash_COLS = MAX_K/(MAX_KP_K/INTERNAL_KP_K_TILE);
  __shared__ T Ash[TILE_X][Ash_COLS];
  const uint C_ELEMS_STORE = N_THREADS * (sizeof(LD_TYPE)/sizeof(T));
  const uint Csh_COLS = MAX_K/(MAX_KP_N/KP_N_TILE);
  const uint Csh_COLS_SIZE = MIN(Csh_COLS, C_ELEMS_STORE);

  uint wid = threadIdx.x/32;
  uint lane = threadIdx.x%32;
  uint blockWarps = blockDim.x/32;
  uint kpK;
  uint kpN;
  uint K;
  uint N;
 
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

  const uint KPK_SPLIT_SIZE = MIN(4, INTERNAL_KP_K_TILE);
  const uint NUM_KPK_SPLITS = MAX(1, INTERNAL_KP_K_TILE/KPK_SPLIT_SIZE);
  const uint ldNumElems = (sizeof(LD_TYPE)/sizeof(T));

  uint external_tile_kp_k = blockIdx.z;
  
  if (KP_N_TILE == MAX_KP_N && INTERNAL_KP_N_TILE == MAX_KP_N && INTERNAL_KP_K_TILE == MAX_KP_K) {
    #ifdef EVAL
      typedef float4 LD_TYPE; 
    #else 
      typedef int4 LD_TYPE; 
    #endif
    const int ldNumElems = sizeof(LD_TYPE)/sizeof(T);
    const int ldSize = MIN(kpN*kpK, ldNumElems);

    for (auto i = threadIdx.x*ldSize; i < (kpN * kpK); i += blockDim.x*ldSize) {
      // kron_fac_sh[i%kpN][i/kpK] = kron_fac[i];
      LD_TYPE a = *(LD_TYPE*)&kron_fac[i];
      T a1[4] = {a.x, a.y, a.z, a.w};
      for (int j = 0; j < ldSize; j++) {
        int idx = i + j;
        kron_fac_sh[idx%kpK][idx/kpK] = a1[j];
      }
    }
  } else {
  }

  
  const uint numKpColMult = MIN(MAX_K/MAX_KP_K, N_THREADS); //Threads executing in parallel to multiply one column of KP with MAX_K row elements of A,
  const uint kpMulblockWarps = MIN(MAX_KP_K, N_THREADS/numKpColMult); //4
  const uint Creg_SIZE = MAX(MIN(Csh_COLS/N_THREADS, 64), 1);
  const uint Creg_Rows = 4; //MAX(MIN(Creg_SIZE, MIN(MAX_K/MAX_KP_K, 8*N_THREADS)/N_THREADS), 1); //Prefer rows > 1 than cols, to use 128-bit stores
  const uint Creg_Cols = 4; //MIN(MAX_KP_K, Creg_SIZE/Creg_Rows);
  
  const uint NUM_INTERNAL_KP_N_TILES = KP_N_TILE/INTERNAL_KP_N_TILE; //1
  // assert(Creg_SIZE == Creg_Cols * Creg_Rows * NUM_INTERNAL_KP_N_TILES);

  register T Creg[Creg_Rows][Creg_Cols];

  uint kpMullane = threadIdx.x%numKpColMult;
  uint kpMulwid = threadIdx.x/numKpColMult;
   //TODO: Names should be different
  const uint kp_col_start_ = (threadIdx.x / ((MAX_K/MAX_KP_K)/Creg_Rows)) * Creg_Cols; //TODO: Fix this, some values of Creg_Rows might not cover all kp_cols
  const uint a_col_start_  = (threadIdx.x % ((MAX_K/MAX_KP_K)/Creg_Rows)) * Creg_Rows;

  for (uint start_row = blockIdx.y * TILE_X; start_row < gridDim.y * TILE_X * N_COARSE_TB; start_row += gridDim.y * TILE_X) {
  // if (start_row == 0 && threadIdx.x == 0) {
  //   printf("Creg_Rows %d Creg_Cols %d\n", Creg_Rows, Creg_Cols);
  // }
  for (uint kp_col_start = kp_col_start_; kp_col_start < MAX_KP_K      ; kp_col_start += N_THREADS * (N_THREADS/ ((MAX_K/MAX_KP_K)/Creg_Rows)) * Creg_Cols) {
  for (uint a_col_start  = a_col_start_ ; a_col_start  < MAX_K/MAX_KP_K; a_col_start  += N_THREADS * (N_THREADS/ ((MAX_K/MAX_KP_K)/Creg_Rows)) * Creg_Rows) {
    // if (start_row == 0 && kp_idx == 0 && threadIdx.x < 64) {
    //   printf("Creg_Rows %d Creg_Cols %d a_col_start %d kp_col_start %d\n", Creg_Rows, Creg_Cols, a_col_start, kp_col_start);
    // }
    #pragma unroll
    for (uint reg_i = 0; reg_i < Creg_Rows; reg_i++) {
      #pragma unroll
      for (uint reg_j = 0; reg_j < Creg_Cols; reg_j++) {
        Creg[reg_i][reg_j] = 0;
      }
    }
  
    for (uint internal_tile_kp_k = 0; internal_tile_kp_k < EXTERNAL_KP_K_TILE; internal_tile_kp_k += INTERNAL_KP_K_TILE) {
      for (uint a_row = 0; a_row < TILE_X; a_row += 1) {
        for (uint a_col = threadIdx.x*ldNumElems; a_col < Ash_COLS; a_col += blockDim.x*ldNumElems) {
          uint tile_k = get_tile_k<MAX_KP_N, KP_N_TILE>();
          LD_TYPE a;
          if (INTERNAL_KP_K_TILE == MAX_KP_K) {
            a = *(LD_TYPE*)&A[(a_row + start_row) * K + (K_EQUALS_VAR ? 0 : tile_k*MAX_K) + a_col];
            // *(LD_TYPE*)&Ash[a_row][a_col] = a;
            T a1[4] = {a.x, a.y, a.z, a.w};
            for (int j = 0; j < ldNumElems; j++) {
              Ash[a_row][a_col + j] = a1[j];
            }
          } else {
            a = *(LD_TYPE*)&A[(a_row + start_row) * K + (K_EQUALS_VAR ? 0 : tile_k*MAX_K) + \
                                      (a_col/INTERNAL_KP_K_TILE)*kpK + external_tile_kp_k * EXTERNAL_KP_K_TILE + internal_tile_kp_k + a_col % INTERNAL_KP_K_TILE];
            *(LD_TYPE*)&Ash[a_row][a_col] = a;
          }
          
          //TODO: Following code stores Ash in a round robin manner. Disabling it for new version
          // T a1[4] = {a.x, a.y, a.z, a.w};
          // for (uint i = 0; i < ldNumElems; i++) {
          //   uint ash_col = a_col + i;
          //   uint lane = ash_col/INTERNAL_KP_K_TILE; // 32
          //   uint kpKlane = lane % INTERNAL_KP_K_TILE; // % 32
           
          //   uint final_col = (ash_col/INTERNAL_KP_K_TILE)*INTERNAL_KP_K_TILE + (ash_col % INTERNAL_KP_K_TILE + kpKlane)%INTERNAL_KP_K_TILE;
          //   Ash[a_row][final_col] = a1[i];
          // }
        }
      }
    
      //TODO: nvcc unrolls this loop, which leads to high register usage
      for (uint internal_tile_kp_n = 0; internal_tile_kp_n < KP_N_TILE; internal_tile_kp_n += INTERNAL_KP_N_TILE) {
        if (!(KP_N_TILE == MAX_KP_N && INTERNAL_KP_N_TILE == MAX_KP_N && INTERNAL_KP_K_TILE == MAX_KP_K)) {
          //Create kpK subwarps and each subwarp loads 0 to INTERNAL_KP_N_TILE elements
          #ifdef EVAL
            typedef float4 LD_TYPE; 
          #else 
            typedef int4 LD_TYPE; 
          #endif
          const uint ldNumElems = sizeof(LD_TYPE)/sizeof(T);
          const uint ldSize = MIN(INTERNAL_KP_N_TILE, ldNumElems);

          for (uint swid = threadIdx.x/(INTERNAL_KP_N_TILE/ldSize); swid < INTERNAL_KP_K_TILE; swid += blockDim.x/(INTERNAL_KP_N_TILE/ldSize)) {
            uint external_tile_kp_n = get_external_tile_kp_n<MAX_KP_N, KP_N_TILE>();
            uint col = external_tile_kp_n*KP_N_TILE + internal_tile_kp_n + (threadIdx.x%(INTERNAL_KP_N_TILE/ldSize))*ldSize;
            uint row = swid;
            // kron_fac_sh[threadIdx.x%INTERNAL_KP_N_TILE][row] = kron_fac[(external_tile_kp_k * EXTERNAL_KP_K_TILE + internal_tile_kp_k + row) * kpN + col];
            LD_TYPE a = *(LD_TYPE*)&kron_fac[(external_tile_kp_k * EXTERNAL_KP_K_TILE + internal_tile_kp_k + row) * kpN + col];
            T a1[4] = {a.x, a.y, a.z, a.w};
            for (uint i = 0; i < ldSize; i++) {
              uint idx = (threadIdx.x%(INTERNAL_KP_N_TILE/ldSize))*ldSize + i%ldSize;
              kron_fac_sh[idx][row] = a1[i];
            }
          }
        }

        __syncthreads();
        
        for (uint a_row = 0; a_row < TILE_X; a_row++) {
          // assert (kp_col_start == 0);
          // #pragma unroll
          {
            const uint MAX_AR_SZ = KPK_SPLIT_SIZE;

            //Load MAX_AR_SZ elements at a time to limit the register usage
            for (uint ar_start_id = 0; ar_start_id < INTERNAL_KP_K_TILE; ar_start_id += MAX_AR_SZ) { //TODO: Shared memory bank conflicts with kpK = 32 and AR_SZ = 16
              register T Ar[Creg_Rows][MAX_AR_SZ];
              register T KPr[MAX_AR_SZ][Creg_Cols];
              //TODO: Following code loads from shared memory in round robin manner. Disabling it for now.
              // uint kpKlane = (lane % numKpColMult) % INTERNAL_KP_K_TILE; //

              // for (uint a_col = kpKlane, i = 0; i < MAX_AR_SZ; i++) { //
              //     Ar[i] = Ash[a_row][(a_col_start+kpMullane)*INTERNAL_KP_K_TILE + (ar_start_id + a_col + i) % INTERNAL_KP_K_TILE];//TODO: Shared memory bank conflicts here with KP_K = 4
              // }
              
              for (uint _a_col = 0; _a_col < Creg_Rows; _a_col++) {
                uint a_col = a_col_start + _a_col;
                for (uint a_elem = 0; a_elem < MAX_AR_SZ; a_elem++)    
                  Ar[_a_col][a_elem] = Ash[a_row][a_col * INTERNAL_KP_K_TILE + ar_start_id + a_elem]; //TODO: Add ar_start_id
              }

              for (uint _kp_col = 0; _kp_col < Creg_Cols; _kp_col++) {
                uint kp_col = kp_col_start + _kp_col;
                for (uint elem = 0; elem < MAX_AR_SZ; elem++)    
                  KPr[elem][_kp_col] = kron_fac_sh[kp_col][ar_start_id + elem]; //TODO: Add ar_start_id
              }

              for (int i = 0; i < Creg_Rows; i++)
                for (int j = 0; j < Creg_Cols; j++)
                  for (int k = 0; k < MAX_AR_SZ; k++)
                    Creg[i][j] += Ar[i][k] * KPr[k][j];
            }
          }
        }
      }
    }
  
    for (uint reg_j = 0; reg_j < Creg_Cols; reg_j++) {
      if (Creg_Rows % 4 == 0) {
        for (uint reg_i = 0; reg_i < Creg_Rows; reg_i += 4) {
          int a_row = 0;
          
          const uint c_row = (a_row + start_row);
          const uint c_col = kp_col_start*(MAX_K/MAX_KP_K) + reg_j*(MAX_K/MAX_KP_K) + a_col_start + reg_i;
          const uint c_idx = c_row * N + c_col;
          // assert(threadIdx.x == c_col);
          // if (kp_idx == 0&& c_row == 0 && c_col < 64)
          //   printf("threadIdx.x %d c_col %d kp_col_start %d a_col_start %d reg_i %d reg_j %d\n", threadIdx.x, c_col, kp_col_start, a_col_start, reg_i, reg_j);
          if (c_col < K) {
            LD_TYPE c = {Creg[reg_i][reg_j], Creg[reg_i+1][reg_j], Creg[reg_i+2][reg_j], Creg[reg_i+3][reg_j]};
            *(LD_TYPE*)&C[c_idx] = c;
          }
        }
      } else {
        assert(false);
      }
    }
    // for (uint reg = 0; reg < Creg_SIZE; reg++) {
    //   uint a_row = reg / (Creg_SIZE/TILE_X);
    //   uint c_row = (a_row + start_row);
    //   uint c_idx;
    //   uint reg_col = reg % (Creg_SIZE/TILE_X);
    //   uint c_col;
    //   uint external_tile_kp_n = get_external_tile_kp_n<MAX_KP_N, KP_N_TILE>();      
    //   //Cannot store N_THREADS values in shared memory. So may be not do it then?
    //   c_col = (reg_col/(Creg_Cols * Creg_Rows)) * (MAX_K/kpK) * INTERNAL_KP_N_TILE  +
    //   ((reg_col/Creg_Cols) % Creg_Rows) * N_THREADS +
    //   (reg_col%Creg_Cols) * (N_THREADS * (MAX_K/kpK)/numKpColMult) + 
    //   threadIdx.x;

    //   if (!K_EQUALS_VAR) {
    //     uint tile_k = get_tile_k<MAX_KP_N, KP_N_TILE>();
    //     c_col = tile_k * (MAX_K/kpK) + 
    //             (c_col/(MAX_K/kpK)) * (K/kpK) +
    //             c_col%(MAX_K/kpK);
    //   }

    //   if (KP_N_TILE != MAX_KP_N) {
    //     c_col += external_tile_kp_n*(K/(MAX_KP_N/KP_N_TILE)); 
    //   }
      
    //   c_idx = c_row * N + c_col;
    //   //TODO: Store to shared memory and then to global memory using vector stores
    //   if (c_col < K) {
    //     C[c_idx] = Creg[reg];
    //   }      
        //Not worth storing in shared memory and then doing 128-bit stores
        // for (uint reg2 = 0; reg2 < Creg_elems_in_sh; reg2++) {
        //   uint reg = reg1 + reg2;
        //   uint store_index = threadIdx.x + reg2 * N_THREADS;

        //   Creg_in_sh[store_index] = Creg[reg];
        // }

        // __syncthreads();
        // // const int ldNumElems = sizeof(T)/sizeof(T);
        // for (uint csh_col = threadIdx.x*ldNumElems; csh_col < Creg_elems_in_sh*N_THREADS; csh_col += N_THREADS*ldNumElems) {
        //   uint c_row = (a_row + start_row);
        //   uint c_idx;
        //   uint external_tile_kp_n = get_external_tile_kp_n<MAX_KP_N, KP_N_TILE>();
        //   uint c_col = csh_col + (reg1/Creg_elems_in_sh)*N_THREADS*Creg_elems_in_sh;

        //   if (K_EQUALS_VAR) {
        //     c_idx = c_row * N  + c_col + external_tile_kp_n*(K/(MAX_KP_N/KP_N_TILE)); //TODO: Fix when external_tile_kp_n > 0
        //   } else {
        //     uint tile_k = get_tile_k<MAX_KP_N, KP_N_TILE>();

        //     c_idx = c_row * N + external_tile_kp_n*(K/(MAX_KP_N/KP_N_TILE)) + tile_k * (MAX_K/kpK) + (c_col/(MAX_K/kpK)) * (K/kpK) + c_col%(MAX_K/kpK);
        //   }
          
        //   *(LD_TYPE*)&C[c_idx] = *(LD_TYPE*)&Creg_in_sh[csh_col];
        // }

    __syncthreads();
  }}}
}

#define N_THREADS 256 
#define KP_N_TILE 128

#ifdef EVAL
    typedef float DATA_TYPE;
  #else
    typedef int DATA_TYPE;
  #endif

#define TILE_X 1

#define K_EQUALS_VAR_KERNELS(N_COARSE_TB, MAX_K, KP_N_K, K_EQUALS_VAR) \
(void*)cuda_gemm<DATA_TYPE,N_THREADS,N_COARSE_TB,TILE_X,MAX_K,KP_N_K,KP_N_K,KP_N_TILE,K_EQUALS_VAR,1>,
  // (void*)cuda_gemm<DATA_TYPE,N_THREADS,N_COARSE_TB,TILE_X,MAX_K,KP_N_K,KP_N_K,KP_N_TILE,K_EQUALS_VAR,0>,

#define KP_N_K_KERNELS(N_COARSE_TB, MAX_K, KP_N_K) \
  K_EQUALS_VAR_KERNELS(N_COARSE_TB, MAX_K, KP_N_K, 0) \
  K_EQUALS_VAR_KERNELS(N_COARSE_TB, MAX_K, KP_N_K, 1)

#define MAX_K_KERNELS(N_COARSE_TB, MAX_K) \
   KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 64)
// KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 2) \
//   KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 4) \
//   KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 8) \
//   KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 16) \
//   KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 32) \
//   KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 64) \
//   KP_N_K_KERNELS(N_COARSE_TB, MAX_K, 128) 


#define COARSE_TB_KERNELS(N_COARSE_TB) \
MAX_K_KERNELS(N_COARSE_TB, 4096) \
  // MAX_K_KERNELS(N_COARSE_TB, 128) \
  // MAX_K_KERNELS(N_COARSE_TB, 256) \
  // MAX_K_KERNELS(N_COARSE_TB, 512) \
  // MAX_K_KERNELS(N_COARSE_TB, 1024) \
  // MAX_K_KERNELS(N_COARSE_TB, 2048) \
  // MAX_K_KERNELS(N_COARSE_TB, 4096) \  
  // MAX_K_KERNELS(N_COARSE_TB, 8192) \

  // MAX_K_KERNELS(N_COARSE_TB, 16) \
  // MAX_K_KERNELS(N_COARSE_TB, 32) \
  // MAX_K_KERNELS(N_COARSE_TB, 64) \
  
#define MAX_K 4096
#define MIN_K 4096
#define MIN_KP_K 64
#define NUM_MAX_K_KERNELS 1//7
#define NUM_KP_N_K_KERNELS 1//7
#define NUM_COARSE_TB_KERNELS 1
#define NUM_K_EQUALS_VAR 2
#define NUM_KPK_EQUALS_VAR 1

static void* cudaGemmSpecialized[NUM_COARSE_TB_KERNELS][NUM_MAX_K_KERNELS][NUM_KP_N_K_KERNELS][NUM_K_EQUALS_VAR][NUM_KPK_EQUALS_VAR] = {
  // KP_N_K_KERNELS(8, 1024, 32)
    COARSE_TB_KERNELS(1)
    // COARSE_TB_KERNELS(2)
    // COARSE_TB_KERNELS(4)
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
    int N_COARSE_TB = 1; //(M > 100) ? 2 : 1;

    // int idx = (N_COARSE_TB/8)*NUM_MAX_K_KERNELS + (log2(K)-log2(16))*NUM_KP_N_K_KERNELS + (log2(KP_MAT_K[0])-log2(2));
    // printf("idx %d log2(K) %d log2(16) %d\n", idx, log2(K), log2(16));
    // assert(idx < sizeof(cudaGemmSpecialized)/sizeof(void*));
    
    int min_k = min(K, MAX_K);
    int k_equals_var = (min_k == K) ? 1 : 0;
    // if (min_k/KP_MAT_K[0] >= 256) {
    //   //K dimension is very high. Divide it in different threadblocks to have better parallelism
    //   min_k = min_k/KP_MAT_K[0];
    //   k_equals_var = 0;
    // }cudaGemmSpecialized[0][0][0][k_equals_var][1]; //
    cuda_gemm_ty cuda_gemm_func = (cuda_gemm_ty)cudaGemmSpecialized[N_COARSE_TB/2][log2(min_k)-log2(MIN_K)][log2(KP_MAT_K[0])-log2(MIN_KP_K)][k_equals_var][0];
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
    
    // CUDACHECK(cudaDeviceSynchronize());
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
                                          // {10,1024,1024, 10, {2,2,2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2,2,2}},
                                          // {10,1024,1024, 2, {32,32},{32,32}},
                                          {1, 4096, 4096, 2, {64,64},{64,64}},
                                          // {1, 128*128, 128*128, 2, {128,128},{128,128}},
                                          {10,256,256, 2, {16,16},{16,16}},
                                          // // {10,256,256, 2, {16,16},{16,16}},
                                          // {10,512,512, 3, {8,8,8},{8,8,8}},
                                          // {10,256,256, 4, {4,4,4,4},{4,4,4,4}},
                                          // {10,1024,1024, 5, {4,4,4,4,4},{4,4,4,4,4}},
                                          {4,4096,4096, 6, {4,4,4,4,4,4},{4,4,4,4,4,4}},
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
