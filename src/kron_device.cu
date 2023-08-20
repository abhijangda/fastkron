// #define C_IN_SHMEM
template<uint MaxKronCols, uint MaxTileSizeKronCols> __device__ uint get_tile_k() {return blockIdx.y/DIVUP(MaxKronCols, MaxTileSizeKronCols);}
template<uint MaxKronCols, uint MaxTileSizeKronCols> __device__ uint get_external_tile_kp_n() {return blockIdx.y%DIVUP(MaxKronCols, MaxTileSizeKronCols);}

__device__ bool isfirstIdx(dim3 idx) {return idx.x == 0 && idx.y == 0 & idx.z == 0;}

__device__ constexpr uint sqrt(uint x) {
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

//Compute x**y
__device__ __host__ uint power(uint x, int y) {
  uint result = 1;
  for (uint i = 0; i < y; i++) {
    result = result * x;
  }
  return result;
}

template<typename VecT, typename ElemT>
__device__ void globalLoadVec(const ElemT* addr, VecT& vec) {
  //Not implemented
}

template<>
__device__ void globalLoadVec(const float* addr, float4& vec) {
  asm ("ld.ca.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(vec.x), "=f"(vec.y), "=f"(vec.z), "=f"(vec.w) : "l"(addr));
}

template<>
__device__ void globalLoadVec(const int* addr, int4& vec) {
  vec = *(int4*)addr;
}

template<>
__device__ void globalLoadVec(const double* addr, double4& vec) {
  vec = *(double4*)addr;
}

template<>
__device__ void globalLoadVec(const float* addr, float& vec) {
  vec = *addr;
}

template<typename VecT, typename ElemT>
__device__ void loadVecToRegs(VecT& vec, ElemT* regs) {
  //Not implemented
}

//Four Element Vectors
template<typename VecT, typename ElemT>
__device__ void load4ElemVecToRegs(VecT& vec, ElemT* regs) {
  regs[0] = vec.x;
  regs[1] = vec.y;
  regs[2] = vec.z;
  regs[3] = vec.w;
}

template<>
__device__ void loadVecToRegs(float4& vec, float* regs) {
  load4ElemVecToRegs(vec, regs);
}

template<>
__device__ void loadVecToRegs(int4& vec, int* regs) {
  load4ElemVecToRegs(vec, regs);
}


template<>
__device__ void loadVecToRegs(double4& vec, double* regs) {
  load4ElemVecToRegs(vec, regs);
}

//Two element vectors
template<>
__device__ void loadVecToRegs(double2& vec, double* regs) {
  regs[0] = vec.x;
  regs[1] = vec.y;
}


//Single element
template<>
__device__ void loadVecToRegs(float& vec, float* regs) {
  regs[0] = vec;
}

//Store PTX instructions for each vector type
template<typename ElemT>
__device__ void globalStore4Elems(ElemT* addr, ElemT elem1, ElemT elem2, ElemT elem3, ElemT elem4) {
}

template<>
__device__ void globalStore4Elems(float* addr, float elem1, float elem2, float elem3, float elem4) {
  // asm ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "=f"(elem1), "=f"(elem2), "=f"(elem3), "=f"(elem4));
  float4 vec = {elem1, elem2, elem3, elem4};
  *(float4*)addr = vec;
}

template<>
__device__ void globalStore4Elems(int* addr, int elem1, int elem2, int elem3, int elem4) {
  // asm ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "=f"(elem1), "=f"(elem2), "=f"(elem3), "=f"(elem4));
  int4 vec = {elem1, elem2, elem3, elem4};
  *(int4*)addr = vec;
}

template<>
__device__ void globalStore4Elems(double* addr, double elem1, double elem2, double elem3, double elem4) {
  // asm ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "=f"(elem1), "=f"(elem2), "=f"(elem3), "=f"(elem4));
  double4 vec = {elem1, elem2, elem3, elem4};
  *(double4*)addr = vec;
}

template<typename ElemT>
__device__ void globalStore2Elems(ElemT* addr, ElemT elem1, ElemT elem2) {
}

template<>
__device__ void globalStore2Elems(float* addr, float elem1, float elem2) {
  float2 vec = {elem1, elem2};
  *(float2*)addr = vec;
}

template<>
__device__ void globalStore2Elems(int* addr, int elem1, int elem2) {
  int2 vec = {elem1, elem2};
  *(int2*)addr = vec;
}

template<>
__device__ void globalStore2Elems(double* addr, double elem1, double elem2) {
  double2 vec = {elem1, elem2};
  *(double2*)addr = vec;
}

template<typename ElemT>
__device__ void globalStore1Elems(ElemT* addr, ElemT elem1) {
  *addr = elem1;
}


template<typename ElemT, typename VecT, uint NumThreads>
__global__ void copyXtoUVAX(const uint RowsC,    const uint ColsC,   const uint ColsA,
                            const uint KronRows, const uint KronCols,
                            ElemT * __restrict__ uvaTemp,
                            const uint uvaRows, const uint uvaCols,
                            const ElemT * __restrict__ glA,
                            const uint uvaPart) {
  const uint WarpSize     = 32;
  const uint tid          = threadIdx.x;
  const uint wid          = tid/WarpSize;
  const uint lane         = tid%WarpSize;
  const uint blockWarps   = blockDim.x/WarpSize;
  
  const uint rowA = blockIdx.x;

  for (uint uvaElem = tid; uvaElem < uvaCols; uvaElem += NumThreads) {
    uvaTemp[rowA * uvaCols + uvaElem] = glA[rowA* ColsA + uvaPart * uvaCols + uvaElem];  
  }
}

template<typename ElemT, typename VecT, uint NumThreads>
__global__ void copyUVATempToY(const uint RowsC,    const uint ColsC,   const uint ColsA,
                            const uint KronRows, const uint KronCols,
                            ElemT * __restrict__ uvaTemp,
                            const uint uvaRows, const uint uvaCols,
                            ElemT * __restrict__ glC,
                            const uint uvaPart, const uint batchedKronMuls, const uint startKronIdx) {
  const uint WarpSize     = 32;
  const uint tid          = threadIdx.x;
  const uint wid          = tid/WarpSize;
  const uint lane         = tid%WarpSize;
  const uint blockWarps   = blockDim.x/WarpSize;
  const uint rowA = blockIdx.x;

  for (uint uvaElem = tid; uvaElem < uvaCols; uvaElem += NumThreads) {
    // uint cCol = outerTileKronCol*(MaxColsA/MaxKronRows) + reg_j*(MaxColsA/MaxKronRows) + shVecI;
    // //(0,0,0,0,0,16,16,16)*128 + (0,1,2,3,..16)*128
    // if (!K_EQUALS_VAR) {
    //   uint tile_k = get_tile_k<MaxKronCols, MaxTileSizeKronCols>();
    //   cCol = tile_k * (MaxColsA/kronCols) + 
    //       (cCol/(MaxColsA/kronCols)) * (colsA/kronCols) +
    //       cCol%(MaxColsA/kronCols);
    // }
    
    if (batchedKronMuls == 1) {
      uint cCol = uvaPart * (uvaCols/KronRows) + (uvaElem/(uvaCols/KronRows))*(ColsC/KronRows) + uvaElem%(uvaCols/KronRows);
      glC[rowA * ColsA + cCol] = uvaTemp[rowA * uvaCols + uvaElem];
    } else {
      uint KronRowsPower = power(KronRows, batchedKronMuls);
      
      uint UVAColsRatioKronRowsSquare = (uvaCols/KronRowsPower);
      uint withinP5 = uvaPart * UVAColsRatioKronRowsSquare + 
                      ((uvaElem%(uvaCols/KronRows))/UVAColsRatioKronRowsSquare)*(ColsC/(uvaCols/UVAColsRatioKronRowsSquare)) + 
                      uvaElem % UVAColsRatioKronRowsSquare;
      uint p5Index = (uvaElem/(uvaCols/KronRows))*(ColsA/KronRows);
      uint cCol = p5Index + withinP5; //(uvaElem/(uvaCols/KronRows))*(ColsC/KronRows) + uvaElem%(uvaCols/KronRows);
      glC[rowA * ColsA + cCol] = uvaTemp[rowA * uvaCols + uvaElem];
    }
  }
}

template<typename ElemT, typename VecT, bool K_EQUALS_VAR, uint VecTNumElems>
__device__ __forceinline__ 
void shiftAgToAsh(const bool RowsCModTileIsZero, const uint TileSizeRowsA, 
                  const uint TileSizeColsA, const uint MaxKronRows,
                  const uint TileSizeKronRows, const uint MaxColsA,
                  const uint NumThreads, const uint CRegRows,
                  const uint RowsC, const uint kronCols, const uint colsA,
                  const uint tid, const uint tileKronRow, const uint tileRowA,
                  const uint tile_k, const uint external_tile_kp_k,
                  const ElemT* __restrict__ glA, ElemT* __restrict__ shA) {
  for (uint rowA = 0; rowA < (RowsCModTileIsZero ? TileSizeRowsA : MIN(TileSizeRowsA, RowsC - tileRowA)); rowA += 1) {
    for (uint a_col = tid*VecTNumElems; a_col < TileSizeColsA; a_col += NumThreads*VecTNumElems) {
      const ElemT* addrA;
      VecT  vec;
      ElemT elems[VecTNumElems];

      if (TileSizeKronRows == MaxKronRows) {
        addrA = &glA[(rowA + tileRowA) * colsA + (K_EQUALS_VAR ? 0 : tile_k*MaxColsA) + a_col];
        // *(VecT*)&shA[rowA][a_col] = a;
        // ElemT a1[4] = {a.x, a.y, a.z, a.w};
        // for (int j = 0; j < VecTNumElems; j++) {
        //   shA[rowA][a_col + j] = a1[j];
        // }
      } else {
        addrA = &glA[(rowA + tileRowA) * colsA + (K_EQUALS_VAR ? 0 : tile_k*MaxColsA) + \
                     (a_col/TileSizeKronRows)*kronCols + external_tile_kp_k * TileSizeKronRows + tileKronRow + a_col % TileSizeKronRows];
        // *(VecT*)&shA[rowA][a_col] = a;
      }

      globalLoadVec(addrA, vec);
      loadVecToRegs(vec, elems);

      #pragma unroll
      for (uint i = 0; i < VecTNumElems; i++) {
        uint ash_col = a_col + i;
        uint tileColA = (ash_col/TileSizeKronRows)/CRegRows;
       
        uint final_col = (ash_col/TileSizeKronRows)*TileSizeKronRows + (tileColA + ash_col%TileSizeKronRows)%TileSizeKronRows;
        shA[rowA * TileSizeRowsA + final_col] = elems[i];
      }
    }
  }
}


template<typename ElemT, typename VecT, uint VecTNumElems>
__device__ __forceinline__ 
void tiledDirectFglToFsh(const uint TileSizeKronRows, const uint TileSizeKronCols, const uint NumThreads, const uint external_tile_kp_n, const uint external_tile_kp_k, const uint tileKronRow, const uint kronRows, const uint tid, const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const uint loadInstr = MIN(TileSizeKronCols, VecTNumElems);
  //Create kronCols subwarps and each subwarp loads 0 to TileSizeKronRows elements
  for (uint swid = tid/(TileSizeKronCols/loadInstr); swid < TileSizeKronRows; swid += NumThreads/(TileSizeKronCols/loadInstr)) {
    VecT  vec;
    ElemT elems[VecTNumElems];

    const uint col = external_tile_kp_n*TileSizeKronCols + (tid%(TileSizeKronCols/loadInstr))*loadInstr;
    const uint row = swid;
    // shKronMats[tid%TileSizeKronRows][row] = glKronMats[(external_tile_kp_k * TileSizeKronCols + tileKronRow + row) * kronRows + col];

    globalLoadVec(&Fgl[(external_tile_kp_k * TileSizeKronRows + tileKronRow + row) * kronRows + col], vec);
    loadVecToRegs(vec, elems);

    #pragma unroll
    for (uint e = 0; e < loadInstr; e++) {
      uint linearIdx = (tid%(TileSizeKronCols/loadInstr))*loadInstr + e;
      Fsh[row * TileSizeKronCols + linearIdx] = elems[e];
    }
  }
}

template<typename ElemT, typename VecT, uint VecTNumElems>
__device__ __forceinline__ 
void fullDirectFglToFsh(const uint TileSizeKronRows, const uint TileSizeKronCols, const uint NumThreads, const uint MaxKronRows, const uint kronRows, const uint kronCols, const uint tid, const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const uint loadInstr = MIN(kronRows*kronCols, VecTNumElems);

  for (uint eIdx = tid*loadInstr; eIdx < kronRows*kronCols; eIdx += blockDim.x*loadInstr) {
    ElemT regElems[VecTNumElems];
    VecT vec;

    vec = *(VecT*)&Fgl[eIdx];
    loadVecToRegs(vec, regElems);

    #pragma unroll
    for (uint vecElem = 0; vecElem < loadInstr; vecElem++) {
      uint idx = eIdx + vecElem;
      Fsh[(idx/MaxKronRows) * TileSizeKronCols + idx%MaxKronRows] = regElems[vecElem];
    }
  }
}

//KP_N is KronCols
//KP_K is KronRows
// __launch_bounds__(NumThreads)
template<typename ElemT, typename VecT, uint NumThreads, RowParallelismTy RowParallelism, uint TileSizeRowsA, 
         bool RowsCModTileIsZero, uint MaxColsA, uint MaxKronCols, uint MaxKronRows, uint TileSizeKronCols, uint K_EQUALS_VAR,
         uint KPK_EQUALS_VAR, uint CRegRows, uint CRegCols, uint SharedTileKronRows>
__global__ void kronGemmKernel(const uint RowsC,    const uint ColsC,   const uint ColsA,
                               const uint KronRows, const uint KronCols,
                               const ElemT * __restrict__ glA, 
                               const ElemT * __restrict__ glKronMats, 
                               ElemT       * __restrict__ glC,
                               const uint kp_idx) {
  
  const uint WarpSize     = 32;
  const uint tid          = threadIdx.x;
  // const uint wid          = tid/WarpSize;
  // const uint lane         = tid%WarpSize;
  // const uint blockWarps   = blockDim.x/WarpSize;
  const uint VecTNumElems = (sizeof(VecT)/sizeof(ElemT));

  const uint TileSizeKronRows    = MIN(SharedTileKronRows,  MaxKronRows);
  const uint TileSizeColsA       = MaxColsA/(MaxKronRows/TileSizeKronRows);
  
  // const uint CRegSize = MAX((MaxColsA/(MaxKronCols/MaxTileSizeKronCols))/NumThreads, 1);
  // const uint CRegRows = MIN(8, MAX(sqrt(CRegSize), 1));
  // const uint CRegCols = MIN(MaxKronRows, MIN(8, CRegSize/CRegRows));
  
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
  const uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, TileSizeKronCols>();

  constexpr uint wSz = ((MaxColsA/MaxKronRows)/CRegRows);

  const uint kp_col_start_ = (tid / wSz) * CRegCols; 
  const uint a_col_start_  = (tid % wSz) * CRegRows; 

  if (TileSizeKronCols == MaxKronCols && TileSizeKronRows == MaxKronRows) {
    fullDirectFglToFsh<ElemT, VecT, VecTNumElems>(TileSizeKronRows, TileSizeKronCols, NumThreads, MaxKronRows, kronRows, 
    kronCols, tid, glKronMats, &shKronMats[0][0]);
  }

  const uint tileRowA  = blockIdx.x * TileSizeRowsA;
  {
  const uint outerTileKronCol =  kp_col_start_;
  {
  const uint tileColA    =  a_col_start_ ;
  {
    #pragma unroll
    for (uint r = 0; r < TileSizeRowsA; r++) {
    #pragma unroll
    for (uint i = 0; i < CRegRows;      i++) {
    #pragma unroll
    for (uint j = 0; j < CRegCols;      j++) {
      regC[r][i][j] = 0;
    }}}

    for (uint tileKronRow = 0; tileKronRow < kronRows; tileKronRow += TileSizeKronRows) {
      uint tile_k = get_tile_k<MaxKronCols, TileSizeKronCols>();

      shiftAgToAsh<ElemT, VecT, K_EQUALS_VAR, VecTNumElems>(RowsCModTileIsZero, TileSizeRowsA, TileSizeColsA, 
                                                            MaxKronRows, TileSizeKronRows, MaxColsA, NumThreads, CRegRows, RowsC,
                                                            kronCols, colsA, tid, tileKronRow, tileRowA, 
                                                            tile_k, external_tile_kp_k, glA, &shA[0][0]);
    
      if (!(TileSizeKronCols == MaxKronCols && TileSizeKronRows == MaxKronRows)) {
        tiledDirectFglToFsh<ElemT, VecT, VecTNumElems>(TileSizeKronRows, TileSizeKronCols, 
                                                       NumThreads, external_tile_kp_n,
                                                       external_tile_kp_k, tileKronRow, 
                                                       kronRows, tid, glKronMats, 
                                                       &shKronMats[0][0]);
      }

      __syncthreads();

      //Load RegTileSizeACols elements at a time to limit the register usage
      for (uint regTileACol = 0; regTileACol < TileSizeKronRows; regTileACol += RegTileSizeACols) {
        register ElemT Ar[TileSizeRowsA][CRegRows][RegTileSizeACols];
        register ElemT KPr[RegTileSizeACols][CRegCols];

        uint round_start = (tileColA / CRegRows)%TileSizeKronRows;

        #pragma unroll
        for (uint rowA = 0; rowA < TileSizeRowsA; rowA++) {
        if (RowsCModTileIsZero || (TileSizeRowsA > 1 && rowA < RowsC - tileRowA)) {
          #pragma unroll
          for (uint rowC = 0; rowC < CRegRows; rowC++) {
            uint shACol = tileColA + rowC;
            #pragma unroll
            for (uint colC = 0; colC < RegTileSizeACols; colC++)
              Ar[rowA][rowC][colC] = shA[rowA][shACol * TileSizeKronRows + (regTileACol + colC + round_start)%TileSizeKronRows];
        }}}
        
        #pragma unroll
        for (uint colC = 0; colC < CRegCols; colC++) {
          uint shKronCol = outerTileKronCol + colC;//TODO: Should outerTileKronCol be here?
          #pragma unroll
          for (uint elem = 0; elem < RegTileSizeACols; elem++)    
            KPr[elem][colC] = shKronMats[regTileACol + elem][shKronCol];
        }

        #pragma unroll
        for (uint rowA = 0; rowA < TileSizeRowsA; rowA++)
        if (RowsCModTileIsZero || (TileSizeRowsA > 1 && rowA < RowsC - tileRowA)) 
        {
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
      if (RowsCModTileIsZero || (TileSizeRowsA > 1 && rowA < RowsC - tileRowA)) {
        #pragma unroll
        for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
          //Three least significant bits of CRegRows can be either 4, 2, or 1
          constexpr uint vecTyNumElems = CRegRows & (8 - 1);
  #ifndef EVAL
          if (vecTyNumElems != 4 && vecTyNumElems != 2 && vecTyNumElems != 1)
            printf("Invalid vecTyNumElems %d\n", vecTyNumElems);
  #endif
          for (uint reg_i = 0; reg_i < CRegRows; reg_i += vecTyNumElems) {
            if (vecTyNumElems > 1) {
              shA[0][tid * vecTyNumElems] = regC[rowA][reg_i][reg_j];
              shA[0][tid * vecTyNumElems+1] = regC[rowA][reg_i+1][reg_j];
              if (vecTyNumElems > 2) {
                shA[0][tid * vecTyNumElems+2] = regC[rowA][reg_i+2][reg_j];
                shA[0][tid * vecTyNumElems+3] = regC[rowA][reg_i+3][reg_j];
              }
              
              __syncwarp();
              for (uint shVecI = tid%wSz; shVecI < vecTyNumElems*wSz; shVecI += wSz) {
                const uint cRow = rowA + tileRowA;
                uint cCol = outerTileKronCol*(MaxColsA/MaxKronRows) + reg_j*(MaxColsA/MaxKronRows) + shVecI;

                if (!K_EQUALS_VAR) {
                  uint tile_k = get_tile_k<MaxKronCols, TileSizeKronCols>();
                  cCol = tile_k * (MaxColsA/kronCols) + 
                      (cCol/(MaxColsA/kronCols)) * (colsA/kronCols) +
                      cCol%(MaxColsA/kronCols);
                }
                if (TileSizeKronCols != MaxKronCols) {
                  uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, TileSizeKronCols>();
                  cCol += external_tile_kp_n*(colsA/(MaxKronCols/TileSizeKronCols)); 
                }

                const uint cIdx = cRow * colsC + cCol;
                if (cCol < colsA) {
                  glC[cIdx] = shA[0][(tid/wSz)*wSz*vecTyNumElems + shVecI];
                }
              }
              __syncwarp();
            } else {
              const uint cRow = (rowA + tileRowA);
              uint cCol = outerTileKronCol*(MaxColsA/MaxKronRows) + reg_j*(MaxColsA/MaxKronRows) + tileColA + reg_i;
              if (!K_EQUALS_VAR) {
                uint tile_k = get_tile_k<MaxKronCols, TileSizeKronCols>();
                cCol = tile_k * (MaxColsA/kronCols) + 
                    (cCol/(MaxColsA/kronCols)) * (colsA/kronCols) +
                    cCol%(MaxColsA/kronCols);
              }
              if (TileSizeKronCols != MaxKronCols) {
                uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, TileSizeKronCols>();
                cCol += external_tile_kp_n*(colsA/(MaxKronCols/TileSizeKronCols)); 
              }
              const uint cIdx = cRow * colsC + cCol;
              // assert(tid == cCol);
              // if (kp_idx == 0&& cRow == 0 && cCol < 64)
              //   printf("tid %d cCol %d outerTileKronCol %d tileColA %d reg_i %d reg_j %d\n", tid, cCol, outerTileKronCol, tileColA, reg_i, reg_j);
              if (cCol < colsA) {
                switch (vecTyNumElems) {
                  case 4:
                    globalStore4Elems(&glC[cIdx], regC[rowA][reg_i][reg_j], regC[rowA][reg_i+1][reg_j], regC[rowA][reg_i+2][reg_j], regC[rowA][reg_i+3][reg_j]);
                  case 2:
                    globalStore2Elems(&glC[cIdx], regC[rowA][reg_i][reg_j], regC[rowA][reg_i+1][reg_j]);
                  case 1:
                    globalStore1Elems(&glC[cIdx], regC[rowA][reg_i][reg_j]);
                }
              }
            }
          }
        }
      }
    }

    __syncthreads();
  }}}
}