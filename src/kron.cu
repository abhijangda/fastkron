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

// #define C_IN_SHMEM
template<uint MAX_KP_N, uint KP_N_TILE> __device__ uint get_tile_k() {return blockIdx.x/DIVUP(MAX_KP_N, KP_N_TILE);}
template<uint MAX_KP_N, uint KP_N_TILE> __device__ uint get_external_tile_kp_n() {return blockIdx.x%DIVUP(MAX_KP_N, KP_N_TILE);}

__device__ bool isfirstIdx(dim3 idx) {return idx.x == 0 && idx.y == 0 & idx.z == 0;}

__device__ constexpr uint uint_squareroot(uint x)
{
  switch (x) {
    case 1:
      return 1;
    
    case 2:
      return 2;
    
    case 4:
      return 2;
    
    case 8:
      return 4;
    
    case 16:
      return 4;
    
    case 32:
      return 8;
    
    case 64:
      return 8;
    
    default:
      return 1;
  }
}

template<typename VecT, typename T>
__device__ void loadVecToRegs(VecT& vec, T* regs) {
  //Not implemented
}

template<>
__device__ void loadVecToRegs(float4& vec, float* regs) {
  regs[0] = vec.x;
  regs[1] = vec.y;
  regs[2] = vec.z;
  regs[3] = vec.w;
}

template<>
__device__ void loadVecToRegs(int4& vec, int* regs) {
  regs[0] = vec.x;
  regs[1] = vec.y;
  regs[2] = vec.z;
  regs[3] = vec.w;
}


template<>
__device__ void loadVecToRegs(double4& vec, double* regs) {
  regs[0] = vec.x;
  regs[1] = vec.y;
  regs[2] = vec.z;
  regs[3] = vec.w;
}

template<>
__device__ void loadVecToRegs(double2& vec, double* regs) {
  regs[0] = vec.x;
  regs[1] = vec.y;
}

// __launch_bounds__(N_THREADS)
template<typename T, typename VecT, uint N_THREADS, uint N_COARSE_TB, uint TILE_X, uint MAX_K, uint MAX_KP_N, uint MAX_KP_K, uint KP_N_TILE_, uint K_EQUALS_VAR, uint KPK_EQUALS_VAR>
__global__ void cuda_gemm(uint M, uint NVar, uint KVar, const T * __restrict__ A, const T * __restrict__ kron_fac, T * __restrict__ C, uint kpNVar, uint kpKVar, uint kp_idx) {
  const uint KP_N_TILE = MIN(KP_N_TILE_, MAX_KP_N);
  const uint NUM_KP_N_TILES = MAX_KP_N/KP_N_TILE;
  const uint INTERNAL_KP_N_TILE = MIN(128, KP_N_TILE);
  const uint EXTERNAL_KP_K_TILE = MIN(EXTERNAL_KP_K_TILE_, MAX_KP_K);
  const uint INTERNAL_KP_K_TILE = MIN(32, EXTERNAL_KP_K_TILE);

  // printf("MAX_K %d MAX_KP_N %d MAX_KP_K %d KP_N_TILE_ %d\n", MAX_K, MAX_KP_N, MAX_KP_K, KP_N_TILE_);

  __shared__ T kron_fac_sh[INTERNAL_KP_K_TILE][INTERNAL_KP_N_TILE];
  const uint Ash_COLS = MAX_K/(MAX_KP_K/INTERNAL_KP_K_TILE);
  __shared__ T Ash[TILE_X][Ash_COLS];
  const uint C_ELEMS_STORE = N_THREADS * (sizeof(VecT)/sizeof(T));
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

  const uint KPK_SPLIT_SIZE = MIN(16, INTERNAL_KP_K_TILE);
  const uint NUM_KPK_SPLITS = MAX(1, INTERNAL_KP_K_TILE/KPK_SPLIT_SIZE);
  const uint ldNumElems = (sizeof(VecT)/sizeof(T));

  uint external_tile_kp_k = blockIdx.z;
  
  if (KP_N_TILE == MAX_KP_N && INTERNAL_KP_N_TILE == MAX_KP_N && INTERNAL_KP_K_TILE == MAX_KP_K) {
    const uint ldSize = MIN(kpN*kpK, ldNumElems);

    for (uint i = threadIdx.x*ldSize; i < (kpN * kpK); i += blockDim.x*ldSize) {
      // kron_fac_sh[i%kpN][i/kpK] = kron_fac[i];
      VecT a = *(VecT*)&kron_fac[i];
      T a1[ldNumElems];
      loadVecToRegs(a, a1);
      #pragma unroll
      for (uint j = 0; j < ldSize; j++) {
        uint idx = i + j;
        kron_fac_sh[idx/MAX_KP_K][idx%MAX_KP_K] = a1[j];
      }
    }
  } else {
  }

  const uint MAX_CREG_SIZE = MAX((MAX_K/(MAX_KP_N/KP_N_TILE))/N_THREADS, 1);
  const uint Creg_Rows = MIN(8, MAX(uint_squareroot(MAX_CREG_SIZE), 1)); //MAX(MIN(Creg_SIZE, MIN(MAX_K/MAX_KP_K, 8*N_THREADS)/N_THREADS), 1); //Prefer rows > 4 than cols, to use 128-bit stores
  const uint Creg_Cols = MIN(MAX_KP_K, MIN(8, MAX_CREG_SIZE/Creg_Rows)); //MIN(MAX_KP_K, Creg_SIZE/Creg_Rows);
  
#ifndef EVAL
  __syncthreads();
  if (kp_idx == 0 && isfirstIdx(threadIdx) && isfirstIdx(blockIdx)) {
    printf("Creg_Rows %d Creg_Cols %d\n", Creg_Rows, Creg_Cols);
    // for (int i = 0; i < kpN; i++) 
    //   for (int j = 0; j < kpK; j++)
    //     printf("%lf \n", (double)kron_fac_sh[i][j]);
  }
#endif

  const uint NUM_INTERNAL_KP_N_TILES = KP_N_TILE/INTERNAL_KP_N_TILE; //2
  // assert(Creg_SIZE == Creg_Cols * Creg_Rows * NUM_INTERNAL_KP_N_TILES);

  register T Creg[Creg_Rows][Creg_Cols];

  const uint kp_col_start_ = (threadIdx.x / ((MAX_K/MAX_KP_K)/Creg_Rows)) * Creg_Cols;
  const uint a_col_start_  = (threadIdx.x % ((MAX_K/MAX_KP_K)/Creg_Rows)) * Creg_Rows; 

  for (uint start_row = blockIdx.y * TILE_X; start_row < gridDim.y * TILE_X * N_COARSE_TB; 
       start_row += gridDim.y * TILE_X) {
  // if (start_row == 0 && threadIdx.x == 0) {
  //   printf("Creg_Rows %d Creg_Cols %d\n", Creg_Rows, Creg_Cols);
  // }
  
  for (uint kp_col_start = kp_col_start_; kp_col_start < KP_N_TILE      ; 
       kp_col_start +=             MAX(1, N_THREADS/((MAX_K/MAX_KP_K)/Creg_Rows)) * Creg_Cols) { //TODO: Something missing in the increment
  for (uint a_col_start  = a_col_start_ ; a_col_start  < MAX_K/MAX_KP_K ; 
       a_col_start  += N_THREADS * MAX(1, N_THREADS/((MAX_K/MAX_KP_K)/Creg_Rows)) * Creg_Rows) {
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
          VecT a;
          if (INTERNAL_KP_K_TILE == MAX_KP_K) {
            a = *(VecT*)&A[(a_row + start_row) * K + (K_EQUALS_VAR ? 0 : tile_k*MAX_K) + a_col];
            // *(VecT*)&Ash[a_row][a_col] = a;
            // T a1[4] = {a.x, a.y, a.z, a.w};
            // for (int j = 0; j < ldNumElems; j++) {
            //   Ash[a_row][a_col + j] = a1[j];
            // }
          } else {
            a = *(VecT*)&A[(a_row + start_row) * K + (K_EQUALS_VAR ? 0 : tile_k*MAX_K) + \
                                      (a_col/INTERNAL_KP_K_TILE)*kpK + external_tile_kp_k * EXTERNAL_KP_K_TILE + internal_tile_kp_k + a_col % INTERNAL_KP_K_TILE];
            // *(VecT*)&Ash[a_row][a_col] = a;
          }
          
          T a1[ldNumElems];
          loadVecToRegs(a, a1);

          #pragma unroll
          for (uint i = 0; i < ldNumElems; i++) {
            uint ash_col = a_col + i;
            uint a_col_start = (ash_col/INTERNAL_KP_K_TILE)/Creg_Rows;
           
            uint final_col = (ash_col/INTERNAL_KP_K_TILE)*INTERNAL_KP_K_TILE + (a_col_start + ash_col%INTERNAL_KP_K_TILE)%INTERNAL_KP_K_TILE;
            Ash[a_row][final_col] = a1[i];
          }
        }
      }
    
      //TODO: nvcc unrolls this loop, which leads to high register usage
      for (uint internal_tile_kp_n = 0; internal_tile_kp_n < KP_N_TILE; internal_tile_kp_n += INTERNAL_KP_N_TILE) {
        if (!(KP_N_TILE == MAX_KP_N && INTERNAL_KP_N_TILE == MAX_KP_N && INTERNAL_KP_K_TILE == MAX_KP_K)) {
          //Create kpK subwarps and each subwarp loads 0 to INTERNAL_KP_N_TILE elements
          const uint ldNumElems = sizeof(VecT)/sizeof(T);
          const uint ldSize = MIN(INTERNAL_KP_N_TILE, ldNumElems);

          for (uint swid = threadIdx.x/(INTERNAL_KP_N_TILE/ldSize); swid < INTERNAL_KP_K_TILE; swid += blockDim.x/(INTERNAL_KP_N_TILE/ldSize)) {
            uint external_tile_kp_n = get_external_tile_kp_n<MAX_KP_N, KP_N_TILE>();
            uint col = external_tile_kp_n*KP_N_TILE + internal_tile_kp_n + (threadIdx.x%(INTERNAL_KP_N_TILE/ldSize))*ldSize;
            uint row = swid;
            // kron_fac_sh[threadIdx.x%INTERNAL_KP_N_TILE][row] = kron_fac[(external_tile_kp_k * EXTERNAL_KP_K_TILE + internal_tile_kp_k + row) * kpN + col];
            VecT a = *(VecT*)&kron_fac[(external_tile_kp_k * EXTERNAL_KP_K_TILE + internal_tile_kp_k + row) * kpN + col];
            T a1[ldNumElems];
            loadVecToRegs(a, a1);
            #pragma unroll
            for (uint i = 0; i < ldSize; i++) {
              uint idx = (threadIdx.x%(INTERNAL_KP_N_TILE/ldSize))*ldSize + i%ldSize;
              kron_fac_sh[row][idx] = a1[i];
            }
          }
        }

        __syncthreads();
        
        for (uint a_row = 0; a_row < TILE_X; a_row++) {
          const uint MAX_AR_SZ = MIN(4, KPK_SPLIT_SIZE);

          //Load MAX_AR_SZ elements at a time to limit the register usage
          for (uint ar_start_id = 0; ar_start_id < INTERNAL_KP_K_TILE; ar_start_id += MAX_AR_SZ) {
            register T Ar[Creg_Rows][MAX_AR_SZ];
            register T KPr[MAX_AR_SZ][Creg_Cols];

            uint round_start = (a_col_start / Creg_Rows)%INTERNAL_KP_K_TILE;

            #pragma unroll
            for (uint _a_col = 0; _a_col < Creg_Rows; _a_col++) {
              uint a_col = a_col_start + _a_col;
              for (uint a_elem = 0; a_elem < MAX_AR_SZ; a_elem++)    
                Ar[_a_col][a_elem] = Ash[a_row][a_col * INTERNAL_KP_K_TILE + (ar_start_id + a_elem + round_start)%INTERNAL_KP_K_TILE]; 
            }
            
            #pragma unroll
            for (uint _kp_col = 0; _kp_col < Creg_Cols; _kp_col++) {
              uint kp_col = kp_col_start + _kp_col;
              for (uint elem = 0; elem < MAX_AR_SZ; elem++)    
                KPr[elem][_kp_col] = kron_fac_sh[ar_start_id + elem][kp_col];
            }

            #pragma unroll
            for (int i = 0; i < Creg_Rows; i++)
              #pragma unroll
              for (int j = 0; j < Creg_Cols; j++)
                #pragma unroll
                for (int k = 0; k < MAX_AR_SZ; k++)
                  Creg[i][j] += Ar[i][k] * KPr[k][j];
          }
        }
      }

      __syncthreads();
    }

    for (uint reg_j = 0; reg_j < Creg_Cols; reg_j++) {
      if (Creg_Rows % 4 == 0) {
        for (uint reg_i = 0; reg_i < Creg_Rows; reg_i += 4) {
          int a_row = 0;
          
          const uint c_row = (a_row + start_row);
          uint c_col = kp_col_start*(MAX_K/MAX_KP_K) + reg_j*(MAX_K/MAX_KP_K) + a_col_start + reg_i;
          if (!K_EQUALS_VAR) {
            uint tile_k = get_tile_k<MAX_KP_N, KP_N_TILE>();
            c_col = tile_k * (MAX_K/kpK) + 
                (c_col/(MAX_K/kpK)) * (K/kpK) +
                c_col%(MAX_K/kpK);
          }
          if (KP_N_TILE != MAX_KP_N) {
            uint external_tile_kp_n = get_external_tile_kp_n<MAX_KP_N, KP_N_TILE>();
            c_col += external_tile_kp_n*(K/(MAX_KP_N/KP_N_TILE)); 
          }
          const uint c_idx = c_row * N + c_col;
          // assert(threadIdx.x == c_col);
          // if (kp_idx == 0&& c_row == 0 && c_col < 64)
          //   printf("threadIdx.x %d c_col %d kp_col_start %d a_col_start %d reg_i %d reg_j %d\n", threadIdx.x, c_col, kp_col_start, a_col_start, reg_i, reg_j);
          if (c_col < K) {
            VecT c = {Creg[reg_i][reg_j], Creg[reg_i+1][reg_j], Creg[reg_i+2][reg_j], Creg[reg_i+3][reg_j]};
            *(VecT*)&C[c_idx] = c;
          }
        }
      } else {
        for (uint reg_i = 0; reg_i < Creg_Rows; reg_i++) {
          int a_row = 0;
            
          const uint c_row = (a_row + start_row);
          uint c_col = kp_col_start*(MAX_K/MAX_KP_K) + reg_j*(MAX_K/MAX_KP_K) + a_col_start + reg_i;
          
          if (!K_EQUALS_VAR) {
            uint tile_k = get_tile_k<MAX_KP_N, KP_N_TILE>();
            c_col = tile_k * (MAX_K/kpK) + 
                (c_col/(MAX_K/kpK)) * (K/kpK) +
                c_col%(MAX_K/kpK);
          }
          if (KP_N_TILE != MAX_KP_N) {
            uint external_tile_kp_n = get_external_tile_kp_n<MAX_KP_N, KP_N_TILE>();
            c_col += external_tile_kp_n*(K/(MAX_KP_N/KP_N_TILE)); 
          }
          const uint c_idx = c_row * N + c_col;
          // assert(threadIdx.x == c_col);
          // if (kp_idx == 0&& c_row == 0 && c_col < 64)
          //   printf("threadIdx.x %d c_col %d kp_col_start %d a_col_start %d reg_i %d reg_j %d\n", threadIdx.x, c_col, kp_col_start, a_col_start, reg_i, reg_j);
          if (c_col < K) {
            C[c_idx] = Creg[reg_i][reg_j];
          }
        }
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
          
        //   *(VecT*)&C[c_idx] = *(VecT*)&Creg_in_sh[csh_col];
        // }

    __syncthreads();
  }}}
}

#define N_THREADS 256 
#define KP_N_TILE 128

#define TILE_X 1

#define K_EQUALS_VAR_KERNELS(T, VecT, N_COARSE_TB, MAX_K, KP_N_K, K_EQUALS_VAR) \
  (void*)cuda_gemm<T, VecT, N_THREADS,N_COARSE_TB,TILE_X,MAX_K,KP_N_K,KP_N_K,KP_N_TILE,K_EQUALS_VAR,1>,
  // (void*)cuda_gemm<DATA_TYPE,N_THREADS,N_COARSE_TB,TILE_X,MAX_K,KP_N_K,KP_N_K,KP_N_TILE,K_EQUALS_VAR,0>,

#define KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, KP_N_K) \
  K_EQUALS_VAR_KERNELS(T, VecT, N_COARSE_TB, MAX_K, KP_N_K, 0) \
  K_EQUALS_VAR_KERNELS(T, VecT, N_COARSE_TB, MAX_K, KP_N_K, 1)

#define MAX_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K) \
KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 2) \
  KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 4) \
  KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 8) \
  KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 16) \
  KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 32) \
  KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 64) \
//   KP_N_K_KERNELS(T, VecT, N_COARSE_TB, MAX_K, 128) 

#define COARSE_TB_KERNELS(T, VecT, N_COARSE_TB) \
  MAX_K_KERNELS(T, VecT, N_COARSE_TB, 16) \
  MAX_K_KERNELS(T, VecT, N_COARSE_TB, 32) \
  MAX_K_KERNELS(T, VecT, N_COARSE_TB, 64) \
  MAX_K_KERNELS(T, VecT, N_COARSE_TB, 128) \
  MAX_K_KERNELS(T, VecT, N_COARSE_TB, 256) \
  MAX_K_KERNELS(T, VecT, N_COARSE_TB, 512) \
  MAX_K_KERNELS(T, VecT, N_COARSE_TB, 1024) \
  MAX_K_KERNELS(T, VecT, N_COARSE_TB, 2048) \
  MAX_K_KERNELS(T, VecT, N_COARSE_TB, 4096) \
  // MAX_K_KERNELS(T, VecT, N_COARSE_TB, 8192) \
  // MAX_K_KERNELS(T, VecT, N_COARSE_TB, 16384) 

#define TYPE_KERNELS(T, VecT) \
  COARSE_TB_KERNELS(T, VecT, 1)

//Two type kernels float/float4 and int/int4

#define NUM_TYPE_KERNELS 3
#define MIN_K 16
#define MAX_K 4096
#define NUM_MAX_K_KERNELS (log2(MAX_K)-log2(MIN_K) + 1)

#define MIN_KP_K 2
#define MAX_KP_K 64
#define NUM_KP_N_K_KERNELS (log2(MAX_KP_K)-log2(MIN_KP_K) + 1)

#define NUM_COARSE_TB_KERNELS 1
#define NUM_K_EQUALS_VAR 2
#define NUM_KPK_EQUALS_VAR 1

static void* KronGemmKernels[NUM_TYPE_KERNELS][NUM_COARSE_TB_KERNELS][NUM_MAX_K_KERNELS][NUM_KP_N_K_KERNELS][NUM_K_EQUALS_VAR][NUM_KPK_EQUALS_VAR] = {
  // KP_N_K_KERNELS(8, 1024, 32)
  TYPE_KERNELS(float, float4)
  TYPE_KERNELS(int, int4)
  TYPE_KERNELS(double, double4)
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
static bool checkKronMatrixSizes(uint NumKronMats, uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[]) {
  int n=1,k=1;
  for (int i = 0; i < NumKronMats; i++) {
    k *= KronMatRows[i];
    n *= KronMatCols[i];
  }
  if (n != N || k != K) {
    printf("Invalid KP Factors Sizes %d != %d, %d != %d\n", n, N, k, K);
    return false;
  }

  return true;
}

//Launch cuda kernels
template<typename T, typename VecT>
cudaError_t generalKronGemm(const uint NumKronMats, T* kronGemmResults[], T* x, T* kronMats[], T** kronGemmResult,
                           uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream)
{
  typedef int (*KronGemmKernel)(int, int, int, T*, T*, T*, int kpNVar, int kpKVar);
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
              (K/min_k) * DIVUP(KronMatCols[0], KP_N_TILE), 
              DIVUP((M/TILE_X), N_COARSE_TB), 
              DIVUP(KronMatRows[0], EXTERNAL_KP_K_TILE_)
           };
    block = {
              N_THREADS, 
              1, 
              1
            };
    
    //Create kernel args;
    void *args[] = {
                    &M, &N, &K, 
                    &prevResult, 
                    (void*)&kronMats[NumKronMats-i-1], 
                    (void*)kronGemmResult, 
                    (void*)&KronMatCols[NumKronMats-i-1], 
                    (void*)&KronMatRows[NumKronMats-i-1], 
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
