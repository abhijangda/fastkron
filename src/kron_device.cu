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

//KP_N is KronCols
//KP_K is KronRows

// __launch_bounds__(NumThreads)
template<typename ElemT, typename VecT, uint NumThreads, uint N_COARSE_TB, uint TileSizeRowsA, uint MaxColsA, uint MaxKronCols, uint MaxKronRows, uint KP_N_TILE_, uint K_EQUALS_VAR, uint KPK_EQUALS_VAR>
__global__ void kronGemmKernel(const uint RowsC,    const uint ColsC,   const uint ColsA,
                               const uint KronRows, const uint KronCols,
                               const ElemT * __restrict__ glA, 
                               const ElemT * __restrict__ glKronMats, 
                               ElemT       * __restrict__ glC,
                               const uint kp_idx) {
  
  const uint WarpSize     = 32;
  const uint tid          = threadIdx.x;
  const uint wid          = tid/WarpSize;
  const uint lane         = tid%WarpSize;
  const uint blockWarps   = blockDim.x/WarpSize;
  const uint VecTNumElems = (sizeof(VecT)/sizeof(ElemT));

  const uint MaxTileSizeKronCols = MIN(KP_N_TILE_,          MaxKronCols);
  const uint MaxTileSizeKronRows = MIN(EXTERNAL_KP_K_TILE_, MaxKronRows);
  const uint TileSizeKronRows    = MIN(32,                 MaxTileSizeKronRows);
  const uint TileSizeKronCols    = MIN(128,                  MaxTileSizeKronCols);
  const uint TileSizeColsA       = MaxColsA/(MaxKronRows/TileSizeKronRows);
  
  const uint CRegSize = MAX((MaxColsA/(MaxKronCols/MaxTileSizeKronCols))/NumThreads, 1);
  const uint CRegRows = MIN(8, MAX(sqrt(CRegSize), 1));
  const uint CRegCols = MIN(MaxKronRows, MIN(8, CRegSize/CRegRows));
  
  register   ElemT regC[TileSizeRowsA][CRegRows][CRegCols];
  __shared__ ElemT shA[TileSizeRowsA][TileSizeColsA];
  __shared__ ElemT shKronMats[TileSizeKronRows][TileSizeKronCols];

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
    kronCols = MaxKronCols;
    kronRows = MaxKronRows;
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

  const uint RegTileSizeACols = MIN(8, TileSizeKronCols);
  
  const uint external_tile_kp_k = blockIdx.z;

  const uint kp_col_start_ = (tid / ((MaxColsA/MaxKronRows)/CRegRows)) * CRegCols;
  const uint a_col_start_  = (tid % ((MaxColsA/MaxKronRows)/CRegRows)) * CRegRows; 

  if (MaxTileSizeKronCols == MaxKronCols && TileSizeKronCols == MaxKronCols && TileSizeKronRows == MaxKronRows) {
    const uint loadInstr = MIN(kronRows*kronCols, VecTNumElems);

    for (uint eIdx = tid*loadInstr; eIdx < kronRows*kronCols; eIdx += blockDim.x*loadInstr) {
      ElemT regElems[VecTNumElems];
      VecT vec;

      vec = *(VecT*)&glKronMats[eIdx];
      loadVecToRegs(vec, regElems);

      #pragma unroll
      for (uint vecElem = 0; vecElem < loadInstr; vecElem++) {
        uint idx = eIdx + vecElem;
        shKronMats[idx/MaxKronRows][idx%MaxKronRows] = regElems[vecElem];
      }
    }
  }

  for (uint tileRowA  = blockIdx.x * TileSizeRowsA;
            tileRowA  < gridDim.x  * TileSizeRowsA * N_COARSE_TB;
            tileRowA += gridDim.x  * TileSizeRowsA) {
  // if (tileRowA == 0 && tid == 0) {
  //   printf("CRegRows %d CRegCols %d\n", CRegRows, CRegCols);
  // }

  for (uint outerTileKronCol =  kp_col_start_;
            outerTileKronCol <  MaxTileSizeKronCols;
            outerTileKronCol += MAX(1, NumThreads/((MaxColsA/MaxKronRows)/CRegRows)) * CRegCols) {

  for (uint tileColA    =  a_col_start_ ;
            tileColA    <  MaxColsA/MaxKronRows;
            tileColA    += NumThreads * MAX(1, NumThreads/((MaxColsA/MaxKronRows)/CRegRows)) * CRegRows) {

    #pragma unroll
    for (uint r = 0; r < TileSizeRowsA; r++) {
    #pragma unroll
    for (uint i = 0; i < CRegRows;      i++) {
    #pragma unroll
    for (uint j = 0; j < CRegCols;      j++) {
      regC[r][i][j] = 0;
    }}}

    for (uint tileKronRow = 0; tileKronRow < MaxTileSizeKronRows; tileKronRow += TileSizeKronRows) {
      for (uint rowA = 0; rowA < TileSizeRowsA; rowA += 1) {
        for (uint a_col = tid*VecTNumElems; a_col < TileSizeColsA; a_col += blockDim.x*VecTNumElems) {
          uint tile_k = get_tile_k<MaxKronCols, MaxTileSizeKronRows>();
          VecT a;
          if (TileSizeKronRows == MaxKronRows) {
            a = *(VecT*)&glA[(rowA + tileRowA) * colsA + (K_EQUALS_VAR ? 0 : tile_k*MaxColsA) + a_col];
            // *(VecT*)&shA[rowA][a_col] = a;
            // ElemT a1[4] = {a.x, a.y, a.z, a.w};
            // for (int j = 0; j < VecTNumElems; j++) {
            //   shA[rowA][a_col + j] = a1[j];
            // }
          } else {
            a = *(VecT*)&glA[(rowA + tileRowA) * colsA + (K_EQUALS_VAR ? 0 : tile_k*MaxColsA) + \
                           (a_col/TileSizeKronRows)*kronCols + external_tile_kp_k * MaxTileSizeKronCols + tileKronRow + a_col % TileSizeKronRows];
            // *(VecT*)&shA[rowA][a_col] = a;
          }
          
          ElemT a1[VecTNumElems];
          loadVecToRegs(a, a1);

          #pragma unroll
          for (uint i = 0; i < VecTNumElems; i++) {
            uint ash_col = a_col + i;
            uint tileColA = (ash_col/TileSizeKronRows)/CRegRows;
           
            uint final_col = (ash_col/TileSizeKronRows)*TileSizeKronRows + (tileColA + ash_col%TileSizeKronRows)%TileSizeKronRows;
            shA[rowA][final_col] = a1[i];
          }
        }
      }
    
      //TODO: nvcc unrolls this loop, which leads to high register usage
      for (uint tileKronCol = 0; tileKronCol < MaxTileSizeKronCols; tileKronCol += TileSizeKronCols) {
        if (!(MaxTileSizeKronCols == MaxKronCols && TileSizeKronCols == MaxKronCols && TileSizeKronRows == MaxKronRows)) {
          //Create kronCols subwarps and each subwarp loads 0 to TileSizeKronRows elements
          const uint loadInstr = MIN(TileSizeKronCols, VecTNumElems);

          for (uint swid = tid/(TileSizeKronCols/loadInstr); swid < TileSizeKronRows; swid += blockDim.x/(TileSizeKronCols/loadInstr)) {
            VecT  vec;
            ElemT elems[VecTNumElems];

            const uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, MaxTileSizeKronCols>();
            const uint col = external_tile_kp_n*MaxTileSizeKronCols + tileKronCol + (tid%(TileSizeKronCols/loadInstr))*loadInstr;
            const uint row = swid;
            // shKronMats[tid%TileSizeKronRows][row] = glKronMats[(external_tile_kp_k * MaxTileSizeKronCols + tileKronRow + row) * kronRows + col];

            vec = *(VecT*)&glKronMats[(external_tile_kp_k * MaxTileSizeKronRows + tileKronRow + row) * kronRows + col];
            loadVecToRegs(vec, elems);

            #pragma unroll
            for (uint e = 0; e < loadInstr; e++) {
              uint linearIdx = (tid%(TileSizeKronCols/loadInstr))*loadInstr + e;
              shKronMats[row][linearIdx] = elems[e];
            }
          }
        }

        __syncthreads();

        //Load RegTileSizeACols elements at a time to limit the register usage
        for (uint regTileACol = 0; regTileACol < TileSizeKronRows; regTileACol += RegTileSizeACols) {
          register ElemT Ar[TileSizeRowsA][CRegRows][RegTileSizeACols];
          register ElemT KPr[RegTileSizeACols][CRegCols];

          uint round_start = (tileColA / CRegRows)%TileSizeKronRows;

          #pragma unroll
          for (uint rowA = 0; rowA < TileSizeRowsA; rowA++) {
          #pragma unroll
          for (uint rowC = 0; rowC < CRegRows; rowC++) {
              uint shACol = tileColA + rowC;
              #pragma unroll
              for (uint colC = 0; colC < RegTileSizeACols; colC++)
                Ar[rowA][rowC][colC] = shA[rowA][shACol * TileSizeKronRows + (regTileACol + colC + round_start)%TileSizeKronRows];
          }}
          
          #pragma unroll
          for (uint colC = 0; colC < CRegCols; colC++) {
            uint shKronCol = outerTileKronCol + colC;//TODO: Should outerTileKronCol be here?
            #pragma unroll
            for (uint elem = 0; elem < RegTileSizeACols; elem++)    
              KPr[elem][colC] = shKronMats[regTileACol + elem][shKronCol];
          }

          #pragma unroll
          for (uint rowA = 0; rowA < TileSizeRowsA; rowA++)
          #pragma unroll
          for (uint i = 0;    i < CRegRows;         i++)
          #pragma unroll
          for (uint j = 0;    j < CRegCols;         j++)
          #pragma unroll
          for (uint k = 0;    k < RegTileSizeACols; k++)
            regC[rowA][i][j] += Ar[rowA][i][k] * KPr[k][j];
        }
      }

      __syncthreads();
    }

    #pragma unroll
    for (int rowA = 0; rowA < TileSizeRowsA; rowA++) {
      #pragma unroll
      for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
        if (CRegRows % 4 == 0) {
          for (uint reg_i = 0; reg_i < CRegRows; reg_i += 4) {          
            const uint cRow = (rowA + tileRowA);
            uint cCol = outerTileKronCol*(MaxColsA/MaxKronRows) + reg_j*(MaxColsA/MaxKronRows) + tileColA + reg_i;
            if (!K_EQUALS_VAR) {
              uint tile_k = get_tile_k<MaxKronCols, MaxTileSizeKronCols>();
              cCol = tile_k * (MaxColsA/kronCols) + 
                  (cCol/(MaxColsA/kronCols)) * (colsA/kronCols) +
                  cCol%(MaxColsA/kronCols);
            }
            if (MaxTileSizeKronCols != MaxKronCols) {
              uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, MaxTileSizeKronCols>();
              cCol += external_tile_kp_n*(colsA/(MaxKronCols/MaxTileSizeKronCols)); 
            }
            const uint cIdx = cRow * colsC + cCol;
            // assert(tid == cCol);
            // if (kp_idx == 0&& cRow == 0 && cCol < 64)
            //   printf("tid %d cCol %d outerTileKronCol %d tileColA %d reg_i %d reg_j %d\n", tid, cCol, outerTileKronCol, tileColA, reg_i, reg_j);
            if (cCol < colsA) {
              VecT c = {regC[rowA][reg_i][reg_j], regC[rowA][reg_i+1][reg_j], regC[rowA][reg_i+2][reg_j], regC[rowA][reg_i+3][reg_j]};
              *(VecT*)&glC[cIdx] = c;
            }
          }
        } else {
          for (uint reg_i = 0; reg_i < CRegRows; reg_i++) {            
            const uint cRow = (rowA + tileRowA);
            uint cCol = outerTileKronCol*(MaxColsA/MaxKronRows) + reg_j*(MaxColsA/MaxKronRows) + tileColA + reg_i;
            
            if (!K_EQUALS_VAR) {
              uint tile_k = get_tile_k<MaxKronCols, MaxTileSizeKronCols>();
              cCol = tile_k * (MaxColsA/kronCols) + 
                  (cCol/(MaxColsA/kronCols)) * (colsA/kronCols) +
                  cCol%(MaxColsA/kronCols);
            }
            if (MaxTileSizeKronCols != MaxKronCols) {
              uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, MaxTileSizeKronCols>();
              cCol += external_tile_kp_n*(colsA/(MaxKronCols/MaxTileSizeKronCols));
            }
            const uint cIdx = cRow * colsC + cCol;
            // assert(tid == cCol);
            // if (kp_idx == 0&& cRow == 0 && cCol < 64)
            //   printf("tid %d cCol %d outerTileKronCol %d tileColA %d reg_i %d reg_j %d\n", tid, cCol, outerTileKronCol, tileColA, reg_i, reg_j);
            if (cCol < colsA) {
              glC[cIdx] = regC[rowA][reg_i][reg_j];
            }
          }
        }
      }
    }

    __syncthreads();
  }}}
}