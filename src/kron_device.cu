// #define C_IN_SHMEM
template<uint MAX_KP_N, uint KP_N_TILE> __device__ uint get_tile_k() {return blockIdx.y/DIVUP(MAX_KP_N, KP_N_TILE);}
template<uint MAX_KP_N, uint KP_N_TILE> __device__ uint get_external_tile_kp_n() {return blockIdx.y%DIVUP(MAX_KP_N, KP_N_TILE);}

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

template<>
__device__ void loadVecToRegs(float& vec, float* regs) {
  regs[0] = vec;
}

template<typename VecT, typename T>
__device__ void createVec(VecT& vec, T* regs) {
  //Not implemented
}

template<>
__device__ void createVec(float4& vec, float* regs) {
  vec.x = regs[0];
  vec.y = regs[1];
  vec.z = regs[2];
  vec.w = regs[3];
}

template<>
__device__ void createVec(int4& vec, int* regs) {
  vec.x = regs[0];
  vec.y = regs[1];
  vec.z = regs[2];
  vec.w = regs[3];
}

template<>
__device__ void createVec(float& vec, float* regs) {
  vec = regs[0];
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

  register T Creg[TILE_X][Creg_Rows][Creg_Cols];

  const uint kp_col_start_ = (threadIdx.x / ((MAX_K/MAX_KP_K)/Creg_Rows)) * Creg_Cols;
  const uint a_col_start_  = (threadIdx.x % ((MAX_K/MAX_KP_K)/Creg_Rows)) * Creg_Rows; 

  for (uint start_row = blockIdx.x * TILE_X; start_row < gridDim.x * TILE_X * N_COARSE_TB; 
       start_row += gridDim.x * TILE_X) {
  // if (start_row == 0 && threadIdx.x == 0) {
  //   printf("Creg_Rows %d Creg_Cols %d\n", Creg_Rows, Creg_Cols);
  // }
  
  for (uint kp_col_start = kp_col_start_; kp_col_start < KP_N_TILE      ; 
       kp_col_start +=             MAX(1, N_THREADS/((MAX_K/MAX_KP_K)/Creg_Rows)) * Creg_Cols) {
  for (uint a_col_start  = a_col_start_ ; a_col_start  < MAX_K/MAX_KP_K ; 
       a_col_start  += N_THREADS * MAX(1, N_THREADS/((MAX_K/MAX_KP_K)/Creg_Rows)) * Creg_Rows) {
    #pragma unroll
    for(uint tile_row = 0; tile_row < TILE_X; tile_row++) {
      #pragma unroll
      for (uint reg_i = 0; reg_i < Creg_Rows; reg_i++) {
        #pragma unroll
        for (uint reg_j = 0; reg_j < Creg_Cols; reg_j++) {
          Creg[tile_row][reg_i][reg_j] = 0;
        }
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
        
        const uint MAX_AR_SZ = MIN(8, KPK_SPLIT_SIZE);

        //Load MAX_AR_SZ elements at a time to limit the register usage
        for (uint ar_start_id = 0; ar_start_id < INTERNAL_KP_K_TILE; ar_start_id += MAX_AR_SZ) {
          register T Ar[TILE_X][Creg_Rows][MAX_AR_SZ];
          register T KPr[MAX_AR_SZ][Creg_Cols];

          uint round_start = (a_col_start / Creg_Rows)%INTERNAL_KP_K_TILE;

          #pragma unroll
          for (uint a_row = 0; a_row < TILE_X; a_row++) {
            #pragma unroll
            for (uint _a_col = 0; _a_col < Creg_Rows; _a_col++) {
              uint a_col = a_col_start + _a_col;
              for (uint a_elem = 0; a_elem < MAX_AR_SZ; a_elem++)    
                Ar[a_row][_a_col][a_elem] = Ash[a_row][a_col * INTERNAL_KP_K_TILE + (ar_start_id + a_elem + round_start)%INTERNAL_KP_K_TILE]; 
            }
          }
          
          #pragma unroll
          for (uint _kp_col = 0; _kp_col < Creg_Cols; _kp_col++) {
            uint kp_col = kp_col_start + _kp_col;
            for (uint elem = 0; elem < MAX_AR_SZ; elem++)    
              KPr[elem][_kp_col] = kron_fac_sh[ar_start_id + elem][kp_col];
          }

          #pragma unroll
          for (uint a_row = 0; a_row < TILE_X; a_row++)
            #pragma unroll
            for (int i = 0; i < Creg_Rows; i++)
              #pragma unroll
              for (int j = 0; j < Creg_Cols; j++)
                #pragma unroll
                for (int k = 0; k < MAX_AR_SZ; k++)
                  Creg[a_row][i][j] += Ar[a_row][i][k] * KPr[k][j];
                  // Creg[a_row][i][j] += __fma_rd(Creg[a_row][i][j], Ar[a_row][i][k], KPr[k][j]);
        }
      }

      __syncthreads();
    }
    
    #pragma unroll 
    for (int a_row = 0; a_row < TILE_X; a_row++) {
      #pragma unroll 
      for (uint reg_j = 0; reg_j < Creg_Cols; reg_j++) {
        if (Creg_Rows % ldNumElems == 0) {
          for (uint reg_i = 0; reg_i < Creg_Rows; reg_i += ldNumElems) {
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
              T c[ldNumElems];
              #pragma unroll
              for (uint cj = 0; cj < ldNumElems; cj++) {
                c[cj] = Creg[a_row][reg_i + cj][reg_j];
              }
              VecT cvec;
              createVec(cvec, c);
              *(VecT*)&C[c_idx] = cvec;
            }
          }
        } else {
          for (uint reg_i = 0; reg_i < Creg_Rows; reg_i++) {            
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
              C[c_idx] = Creg[a_row][reg_i][reg_j];
            }
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