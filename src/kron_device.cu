// #define C_IN_SHMEM
template<uint MaxKronCols, uint MaxTileSizeKronRows> __device__ uint get_tile_k() {return blockIdx.y/DIVUP(MaxKronCols, MaxTileSizeKronRows);}
template<uint MaxKronCols, uint MaxTileSizeKronRows> __device__ uint get_external_tile_kp_n() {return blockIdx.y%DIVUP(MaxKronCols, MaxTileSizeKronRows);}

__device__ bool isfirstIdx(dim3 idx) {return idx.x == 0 && idx.y == 0 & idx.z == 0;}

__device__ constexpr uint sqrt(uint x)
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

template<typename VecT, typename ElemT>
__device__ void loadVecToRegs(VecT& vec, ElemT* regs) {
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

// __launch_bounds__(NumThreads)
template<typename ElemT, typename VecT, uint NumThreads, uint N_COARSE_TB, uint TileSizeRowsA, uint MaxColsA, uint MaxKronCols, uint MaxKronRows, uint KP_N_TILE_, uint K_EQUALS_VAR, uint KPK_EQUALS_VAR>
__global__ void kronGemmKernel(const uint RowsC,    const uint ColsC,   const uint ColsA,
                               const uint KronRows, const uint KronCols,
                               const ElemT * __restrict__ glA, 
                               const ElemT * __restrict__ glKronMats, 
                               ElemT       * __restrict__ glC,
                               const uint kp_idx) {
  
  const uint WarpSize   = 32;
  const uint tid        = threadIdx.x;
  const uint wid        = tid/WarpSize;
  const uint lane       = tid%WarpSize;
  const uint blockWarps = blockDim.x/WarpSize;
                                
  const uint MaxTileSizeKronRows = MIN(KP_N_TILE_, MaxKronCols);
  const uint MaxTileSizeKronCols = MIN(EXTERNAL_KP_K_TILE_, MaxKronRows);

  const uint TileSizeKronRows = MIN(128, MaxTileSizeKronRows);
  const uint TileSizeKronCols = MIN(32, MaxTileSizeKronCols);
  const uint TileSizeColsA = MaxColsA/(MaxKronRows/TileSizeKronCols);
  
  const uint CRegSize = MAX((MaxColsA/(MaxKronCols/MaxTileSizeKronRows))/NumThreads, 1);
  const uint CRegRows = MIN(8, MAX(sqrt(CRegSize), 1));
  const uint CRegCols = MIN(MaxKronRows, MIN(8, CRegSize/CRegRows));
  
  register   ElemT Creg[TileSizeRowsA][CRegRows][CRegCols];
  __shared__ ElemT sh[TileSizeRowsA][TileSizeColsA];
  __shared__ ElemT shKronMats[TileSizeKronCols][TileSizeKronRows];

#ifndef EVAL
  __syncthreads();
  if (kp_idx == 0 && isfirstIdx(threadIdx) && isfirstIdx(blockIdx)) {
    printf("CRegRows %d CRegCols %d\n", CRegRows, CRegCols);
    // for (int i = 0; i < kronRows; i++) 
    //   for (int j = 0; j < kronCols; j++)
    //     printf("%lf \n", (double)shKronMats[i][j]);
  }
#endif

  // const uint NUM_INTERNAL_KP_N_TILES = MaxTileSizeKronRows/TileSizeKronRows;
  // assert(Creg_SIZE == CRegCols * CRegRows * NUM_INTERNAL_KP_N_TILES);
  uint kronCols;
  uint kronRows;
  uint colsA;
  uint colsC;
 
  if (KPK_EQUALS_VAR) {
    kronCols = MaxKronRows;
    kronRows = MaxKronCols;
  } else {
    kronCols = KronCols;
    kronRows = KronRows;
  }

  if (K_EQUALS_VAR) {
    colsA = MaxColsA;
    colsC = colsA;
  } else {
    colsA = ColsA;
    colsC = ColsC;
  }

  const uint KPK_SPLIT_SIZE = MIN(16, TileSizeKronCols);
  const uint NUM_KPK_SPLITS = MAX(1, TileSizeKronCols/KPK_SPLIT_SIZE);
  const uint VecTNumElems = (sizeof(VecT)/sizeof(ElemT));

  const uint external_tile_kp_k = blockIdx.z;
  
  if (MaxTileSizeKronRows == MaxKronCols && TileSizeKronRows == MaxKronCols && TileSizeKronCols == MaxKronRows) {
    const uint ldSize = MIN(kronRows*kronCols, VecTNumElems);

    for (uint i = tid*ldSize; i < kronRows * kronCols; i += blockDim.x*ldSize) {
      // shKronMats[i%kronRows][i/kronCols] = glKronMats[i];
      VecT a = *(VecT*)&glKronMats[i];
      ElemT a1[VecTNumElems];
      loadVecToRegs(a, a1);
      #pragma unroll
      for (uint j = 0; j < ldSize; j++) {
        uint idx = i + j;
        shKronMats[idx/MaxKronRows][idx%MaxKronRows] = a1[j];
      }
    }
  } else {
  }

  const uint kp_col_start_ = (tid / ((MaxColsA/MaxKronRows)/CRegRows)) * CRegCols;
  const uint a_col_start_  = (tid % ((MaxColsA/MaxKronRows)/CRegRows)) * CRegRows; 

  for (uint start_row = blockIdx.x * TileSizeRowsA; start_row < gridDim.x * TileSizeRowsA * N_COARSE_TB; 
       start_row += gridDim.x * TileSizeRowsA) {
  // if (start_row == 0 && tid == 0) {
  //   printf("CRegRows %d CRegCols %d\n", CRegRows, CRegCols);
  // }
  
  for (uint kp_col_start = kp_col_start_; kp_col_start < MaxTileSizeKronRows     ; 
       kp_col_start +=              MAX(1, NumThreads/((MaxColsA/MaxKronRows)/CRegRows)) * CRegCols) { 
  for (uint a_col_start  = a_col_start_ ; a_col_start  < MaxColsA/MaxKronRows;
       a_col_start  += NumThreads * MAX(1, NumThreads/((MaxColsA/MaxKronRows)/CRegRows)) * CRegRows) {
    #pragma unroll
    for(uint tile_row = 0; tile_row < TileSizeRowsA; tile_row++) {
      #pragma unroll
      for (uint reg_i = 0; reg_i < CRegRows; reg_i++) {
        #pragma unroll
        for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
          Creg[tile_row][reg_i][reg_j] = 0;
        }
      }
    }
  
    for (uint internal_tile_kp_k = 0; internal_tile_kp_k < MaxTileSizeKronCols; internal_tile_kp_k += TileSizeKronCols) {
      for (uint aRow = 0; aRow < TileSizeRowsA; aRow += 1) {
        for (uint a_col = tid*VecTNumElems; a_col < TileSizeColsA; a_col += blockDim.x*VecTNumElems) {
          uint tile_k = get_tile_k<MaxKronCols, MaxTileSizeKronRows>();
          VecT a;
          if (TileSizeKronCols == MaxKronRows) {
            a = *(VecT*)&glA[(aRow + start_row) * colsA + (K_EQUALS_VAR ? 0 : tile_k*MaxColsA) + a_col];
            // *(VecT*)&shA[aRow][a_col] = a;
            // ElemT a1[4] = {a.x, a.y, a.z, a.w};
            // for (int j = 0; j < VecTNumElems; j++) {
            //   shA[aRow][a_col + j] = a1[j];
            // }
          } else {
            a = *(VecT*)&glA[(aRow + start_row) * colsA + (K_EQUALS_VAR ? 0 : tile_k*MaxColsA) + \
                           (a_col/TileSizeKronCols)*kronCols + external_tile_kp_k * MaxTileSizeKronCols + internal_tile_kp_k + a_col % TileSizeKronCols];
            // *(VecT*)&shA[aRow][a_col] = a;
          }
          
          ElemT a1[VecTNumElems];
          loadVecToRegs(a, a1);

          #pragma unroll
          for (uint i = 0; i < VecTNumElems; i++) {
            uint ash_col = a_col + i;
            uint a_col_start = (ash_col/TileSizeKronCols)/CRegRows;
           
            uint final_col = (ash_col/TileSizeKronCols)*TileSizeKronCols + (a_col_start + ash_col%TileSizeKronCols)%TileSizeKronCols;
            shA[aRow][final_col] = a1[i];
          }
        }
      }
    
      //TODO: nvcc unrolls this loop, which leads to high register usage
      for (uint internal_tile_kp_n = 0; internal_tile_kp_n < MaxTileSizeKronRows; internal_tile_kp_n += TileSizeKronRows) {
        if (!(MaxTileSizeKronRows == MaxKronCols && TileSizeKronRows == MaxKronCols && TileSizeKronCols == MaxKronRows)) {
          //Create kronCols subwarps and each subwarp loads 0 to TileSizeKronRows elements
          const uint VecTNumElems = sizeof(VecT)/sizeof(ElemT);
          const uint ldSize = MIN(TileSizeKronRows, VecTNumElems);

          for (uint swid = tid/(TileSizeKronRows/ldSize); swid < TileSizeKronCols; swid += blockDim.x/(TileSizeKronRows/ldSize)) {
            uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, MaxTileSizeKronRows>();
            uint col = external_tile_kp_n*MaxTileSizeKronRows + internal_tile_kp_n + (tid%(TileSizeKronRows/ldSize))*ldSize;
            uint row = swid;
            // shKronMats[tid%TileSizeKronRows][row] = glKronMats[(external_tile_kp_k * MaxTileSizeKronCols + internal_tile_kp_k + row) * kronRows + col];
            VecT a = *(VecT*)&glKronMats[(external_tile_kp_k * MaxTileSizeKronCols + internal_tile_kp_k + row) * kronRows + col];
            ElemT a1[VecTNumElems];
            loadVecToRegs(a, a1);
            #pragma unroll
            for (uint i = 0; i < ldSize; i++) {
              uint idx = (tid%(TileSizeKronRows/ldSize))*ldSize + i%ldSize;
              shKronMats[row][idx] = a1[i];
            }
          }
        }

        __syncthreads();
        
        const uint MAX_AR_SZ = MIN(8, KPK_SPLIT_SIZE);

        //Load MAX_AR_SZ elements at a time to limit the register usage
        for (uint ar_start_id = 0; ar_start_id < TileSizeKronCols; ar_start_id += MAX_AR_SZ) {
          register ElemT Ar[TileSizeRowsA][CRegRows][MAX_AR_SZ];
          register ElemT KPr[MAX_AR_SZ][CRegCols];

          uint round_start = (a_col_start / CRegRows)%TileSizeKronCols;

          #pragma unroll
          for (uint aRow = 0; aRow < TileSizeRowsA; aRow++) {
            #pragma unroll
            for (uint _a_col = 0; _a_col < CRegRows; _a_col++) {
              uint a_col = a_col_start + _a_col;
              for (uint a_elem = 0; a_elem < MAX_AR_SZ; a_elem++)    
                Ar[aRow][_a_col][a_elem] = shA[aRow][a_col * TileSizeKronCols + (ar_start_id + a_elem + round_start)%TileSizeKronCols]; 
            }
          }
          
          #pragma unroll
          for (uint _kp_col = 0; _kp_col < CRegCols; _kp_col++) {
            uint kp_col = kp_col_start + _kp_col;
            for (uint elem = 0; elem < MAX_AR_SZ; elem++)    
              KPr[elem][_kp_col] = shKronMats[ar_start_id + elem][kp_col];
          }

          #pragma unroll
          for (uint aRow = 0; aRow < TileSizeRowsA; aRow++)
            #pragma unroll
            for (int i = 0; i < CRegRows; i++)
              #pragma unroll
              for (int j = 0; j < CRegCols; j++)
                #pragma unroll
                for (int k = 0; k < MAX_AR_SZ; k++)
                  Creg[aRow][i][j] += Ar[aRow][i][k] * KPr[k][j];
        }
      }

      __syncthreads();
    }
    
    #pragma unroll 
    for (int aRow = 0; aRow < TileSizeRowsA; aRow++) {
      #pragma unroll 
      for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
        if (CRegRows % 4 == 0) {
          for (uint reg_i = 0; reg_i < CRegRows; reg_i += 4) {          
            const uint cRow = (aRow + start_row);
            uint cCol = kp_col_start*(MaxColsA/MaxKronRows) + reg_j*(MaxColsA/MaxKronRows) + a_col_start + reg_i;
            if (!K_EQUALS_VAR) {
              uint tile_k = get_tile_k<MaxKronCols, MaxTileSizeKronRows>();
              cCol = tile_k * (MaxColsA/kronCols) + 
                  (cCol/(MaxColsA/kronCols)) * (colsA/kronCols) +
                  cCol%(MaxColsA/kronCols);
            }
            if (MaxTileSizeKronRows != MaxKronCols) {
              uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, MaxTileSizeKronRows>();
              cCol += external_tile_kp_n*(colsA/(MaxKronCols/MaxTileSizeKronRows)); 
            }
            const uint cIdx = cRow * colsC + cCol;
            // assert(tid == cCol);
            // if (kp_idx == 0&& cRow == 0 && cCol < 64)
            //   printf("tid %d cCol %d kp_col_start %d a_col_start %d reg_i %d reg_j %d\n", tid, cCol, kp_col_start, a_col_start, reg_i, reg_j);
            if (cCol < colsA) {
              VecT c = {Creg[aRow][reg_i][reg_j], Creg[aRow][reg_i+1][reg_j], Creg[aRow][reg_i+2][reg_j], Creg[aRow][reg_i+3][reg_j]};
              *(VecT*)&glC[cIdx] = c;
            }
          }
        } else {
          for (uint reg_i = 0; reg_i < CRegRows; reg_i++) {            
            const uint cRow = (aRow + start_row);
            uint cCol = kp_col_start*(MaxColsA/MaxKronRows) + reg_j*(MaxColsA/MaxKronRows) + a_col_start + reg_i;
            
            if (!K_EQUALS_VAR) {
              uint tile_k = get_tile_k<MaxKronCols, MaxTileSizeKronRows>();
              cCol = tile_k * (MaxColsA/kronCols) + 
                  (cCol/(MaxColsA/kronCols)) * (colsA/kronCols) +
                  cCol%(MaxColsA/kronCols);
            }
            if (MaxTileSizeKronRows != MaxKronCols) {
              uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, MaxTileSizeKronRows>();
              cCol += external_tile_kp_n*(colsA/(MaxKronCols/MaxTileSizeKronRows)); 
            }
            const uint cIdx = cRow * colsC + cCol;
            // assert(tid == cCol);
            // if (kp_idx == 0&& cRow == 0 && cCol < 64)
            //   printf("tid %d cCol %d kp_col_start %d a_col_start %d reg_i %d reg_j %d\n", tid, cCol, kp_col_start, a_col_start, reg_i, reg_j);
            if (cCol < colsA) {
              glC[cIdx] = Creg[aRow][reg_i][reg_j];
            }
          }
        }
      }
    }

    __syncthreads();
  }}}
}