#include<device_functions.cuh>

template<typename ElemT, typename VecT, uint NumThreads, RowParallelismTy RowParallelism, 
         uint TileSizeRowsA, bool RowsCModTileIsZero, uint MaxColsA, uint MaxKronCols, 
         uint MaxKronRows, uint TileSizeKronCols, uint K_EQUALS_VAR,
         uint KPK_EQUALS_VAR, uint CRegRows, uint CRegCols, uint SharedTileKronRows, uint NumFusedKerns>
__launch_bounds__(NumThreads)
__global__ void kronGemmKernel(KernelParams<ElemT, NumFusedKerns> params, 
                               DistributedParams<ElemT> distParams) {
  
  const uint WarpSize     = 32;
  const uint tid          = threadIdx.x;
  // const uint wid          = tid/WarpSize;
  // const uint lane         = tid%WarpSize;
  // const uint blockWarps   = blockDim.x/WarpSize;
  const uint VecTNumElems = (sizeof(VecT)/sizeof(ElemT));

  const uint TileSizeKronRows    = MIN(SharedTileKronRows,  MaxKronRows);
  const uint TileSizeColsA       = MaxColsA/(MaxKronRows/TileSizeKronRows);
  
  static_assert(TileSizeKronCols <= MaxKronCols, "");
  static_assert(NumFusedKerns == 1 || 
                (NumFusedKerns > 1 && TileSizeKronRows >= MaxKronRows && TileSizeKronCols >= MaxKronCols),
                "Invalid tile size params for fusion");
  
  register   ElemT regC[TileSizeRowsA][CRegRows][CRegCols] = {0};
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
    kronCols = params.KronCols[0];
    kronRows = params.KronRows[0];
  }

  if (K_EQUALS_VAR) {
    colsA = MaxColsA;
  } else {
    colsA = params.ColsA;
  }

  colsC = params.ColsC;

  const uint RegTileSizeACols = MIN(8, TileSizeKronCols);
  
  const uint external_tile_kp_k = blockIdx.z;
  const uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, TileSizeKronCols>();
  const uint MaxColsC = (MaxColsA/MaxKronRows)*MaxKronCols;
  constexpr uint wSz = ((MaxColsA/MaxKronRows)/CRegRows); //1024/8 = 128
  // constexpr uint wSz2 = (MaxKronCols/CRegCols); //

  const uint kp_col_start_ = (tid / wSz) * CRegCols; //0 to 8
  const uint a_col_start_  = (tid % wSz) * CRegRows; //0 to 128

  const uint tileRowA         = blockIdx.y * TileSizeRowsA;
  const uint outerTileKronCol = kp_col_start_;
  const uint tileColC         = a_col_start_ ;

  for (uint tileKronRow = 0; tileKronRow < kronRows; tileKronRow += TileSizeKronRows) {
    //Loop iterates only once when NumFusedKerns == 1
    uint tile_k = get_tile_k<MaxKronCols, TileSizeKronCols>();
    shiftAgToAsh<ElemT, VecT, K_EQUALS_VAR, VecTNumElems>(RowsCModTileIsZero, TileSizeRowsA, TileSizeColsA, 
                                                          MaxKronRows, TileSizeKronRows, MaxColsA, NumThreads, CRegRows, params.RowsC,
                                                          kronCols, colsA, tid, 
                                                          tileKronRow, tileRowA, 
                                                          tile_k, external_tile_kp_k, 
                                                          params.glA, &shA[0][0]);

    #pragma unroll
    for (uint fusedFac = 0; fusedFac < NumFusedKerns; fusedFac++) {
      if (NumFusedKerns > 1) {
        #pragma unroll
        for (uint r = 0; r < TileSizeRowsA; r++) {
        #pragma unroll
        for (uint i = 0; i < CRegRows;      i++) {
        #pragma unroll
        for (uint j = 0; j < CRegCols;      j++) {
          regC[r][i][j] = 0;
        }}}
      }
      if (TileSizeKronCols == MaxKronCols && TileSizeKronRows == MaxKronRows) {
        //Optimized to load full factor matrix
        fullDirectFglToFsh<ElemT, VecT, VecTNumElems>(MaxKronRows, MaxKronCols,
                                                      TileSizeKronRows, TileSizeKronCols, 
                                                      NumThreads, kronRows, 
                                                      kronCols, tid, params.glKronMats[fusedFac], &shKronMats[0][0]);
      } else if (!(TileSizeKronCols == MaxKronCols && TileSizeKronRows == MaxKronRows)) {
        tiledDirectFglToFsh<ElemT, VecT, VecTNumElems>(MaxKronRows, MaxKronCols,
                                                       TileSizeKronRows, TileSizeKronCols, 
                                                       NumThreads, external_tile_kp_n,
                                                       external_tile_kp_k, tileKronRow, 
                                                       kronRows, kronCols, tid, params.glKronMats[fusedFac],
                                                       &shKronMats[0][0]);
      } else {
      }
      __syncthreads();

      //Load RegTileSizeACols elements at a time to limit the register usage
      for (uint regTileACol = 0; regTileACol < TileSizeKronRows; regTileACol += RegTileSizeACols) {
        register ElemT Ar[TileSizeRowsA][CRegRows][RegTileSizeACols];
        register ElemT KPr[RegTileSizeACols][CRegCols];
        const uint tileColA = (tileColC);// * MaxColsA)/MaxColsC;
        uint round_start = (tileColA / CRegRows)%TileSizeKronRows;

        #pragma unroll
        for (uint rowA = 0; rowA < TileSizeRowsA; rowA++) {
        if (RowsCModTileIsZero || (TileSizeRowsA > 1 && rowA < params.RowsC - tileRowA)) {
          #pragma unroll
          for (uint rowC = 0; rowC < CRegRows; rowC++) {
            uint shACol = tileColA + rowC;
            #pragma unroll
            for (uint colC = 0; colC < RegTileSizeACols; colC++) {
              // ElemT temp = (params.kp_idx == 2) ? 1 : (params.kp_idx == 1 ? 8 : 64);
              ElemT temp = shA[rowA][shACol * TileSizeKronRows + (regTileACol + colC + round_start)%TileSizeKronRows];
              Ar[rowA][rowC][colC] = temp;
              // if (params.kp_idx == 1 && blockIdx.x == 0 && blockIdx.y == 0 && outerTileKronCol == 0 && tileColC == 8) {
              //   printf("Ar[%d][%d][%d] shACol %d shA[%d][%d] %f\n", 
              //           rowA, rowC, colC, shACol,
              //           rowA, shACol * TileSizeKronRows + (regTileACol + colC + round_start)%TileSizeKronRows,
              //           temp);
              // }
            }
        }}}
        
        #pragma unroll
        for (uint colC = 0; colC < CRegCols; colC++) {
          uint shKronCol = outerTileKronCol + colC;
          #pragma unroll
          for (uint elem = 0; elem < RegTileSizeACols; elem++)    
            KPr[elem][colC] = shKronMats[regTileACol + elem][shKronCol];
        }

        //Matrix Multiply Accumulate
        #pragma unroll
        for (uint rowA = 0; rowA < TileSizeRowsA; rowA++)
        if (RowsCModTileIsZero || (TileSizeRowsA > 1 && rowA < params.RowsC - tileRowA)) {
          #pragma unroll
          for (uint i = 0;    i < CRegRows;         i++)
          #pragma unroll
          for (uint j = 0;    j < CRegCols;         j++)
          #pragma unroll
          for (uint k = 0;    k < RegTileSizeACols; k++) {
            regC[rowA][i][j] += Ar[rowA][i][k] * KPr[k][j];
            // if (Ar[rowA][i][k] == 0.0f) printf("Ar[rowA][%d][%d] %f\n", i,k, Ar[rowA][i][k]);
            // if (tileColC == 0 and outerTileKronCol == 8) {
            //   printf("regC %f KPr[k][j] %f\n", regC[rowA][i][j], KPr[k][j]);
            // }
          }
        }
      }

      __syncthreads();
      if (NumFusedKerns > 1 && fusedFac < NumFusedKerns - 1) {
      //Store C to shared memory using shift method
      for (int rowA = 0; rowA < TileSizeRowsA; rowA++) {
      if (RowsCModTileIsZero || (TileSizeRowsA > 1 && rowA < params.RowsC - tileRowA)) {
        #pragma unroll
        for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
        for (uint reg_i = 0; reg_i < CRegRows; reg_i++) {
          uint cCol = outerTileKronCol*(MaxColsA/MaxKronRows) + reg_j*(MaxColsA/MaxKronRows) + tileColC + reg_i;
          uint tileColC = (cCol/TileSizeKronRows)/CRegRows;
          
          cCol = (cCol/TileSizeKronRows)*TileSizeKronRows + (tileColC + cCol%TileSizeKronRows)%TileSizeKronRows;
          shA[rowA][cCol] = regC[rowA][reg_i][reg_j];
      }}}}
      __syncthreads();
      }
    }
  }

  if (NumFusedKerns > 1) {
    for (uint rowShC = 0; rowShC < TileSizeRowsA; rowShC++) {
    if (RowsCModTileIsZero || (TileSizeRowsA > 1 && rowShC < params.RowsC - tileRowA)) {
      //TODO: Improve below code like in the paper
      //TODO: Can be provided when compiling kernel.
      #ifdef KPK_EQUALS_VAR
      const uint KronRowsPower = iconstpower<kronRows, NumFusedKerns>();
      #else
      const uint KronRowsPower = power(kronRows, NumFusedKerns);
      #endif
      uint UVAColsRatioKronRowsSquare = (TileSizeColsA/KronRowsPower);
      const uint ratio = colsC/KronRowsPower; //(colsC/(TileSizeColsA/UVAColsRatioKronRowsSquare));
      #pragma unroll
      for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
      for (uint reg_i = 0; reg_i < CRegRows; reg_i++) {
        uint colShC = outerTileKronCol*(MaxColsA/MaxKronRows) + reg_j*(MaxColsA/MaxKronRows) + tileColC + reg_i;
        const uint rowC = rowShC + tileRowA;
        //Need below to work well
        //KronRowsPower = 512;
        
        //colsA = 8*8*8*8*8*8;
        uint tile_k = 0;
        if (!K_EQUALS_VAR) {
          tile_k = get_tile_k<MaxKronCols, TileSizeKronCols>();
        }
        uint withinP5 = tile_k * UVAColsRatioKronRowsSquare +
                        ((colShC%(TileSizeColsA/kronRows))/UVAColsRatioKronRowsSquare)*ratio + 
                        colShC%UVAColsRatioKronRowsSquare;
        
        uint p5Index = (colShC/(TileSizeColsA/kronRows))*(colsC/kronRows);
        uint cCol = p5Index + withinP5;
        params.glC[rowC * colsC + cCol] = regC[rowShC][reg_i][reg_j];
    }}}
  }} else {
  #pragma unroll
  for (int rowA = 0; rowA < TileSizeRowsA; rowA++) {
  if (RowsCModTileIsZero || (TileSizeRowsA > 1 && rowA < params.RowsC - tileRowA)) {
    #pragma unroll
    for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
    //Three least significant bits of CRegRows can be either 4, 2, or 1
    constexpr uint vecTyNumElems = CRegRows & (8 - 1);
#ifndef EVAL
    if (vecTyNumElems != 4 && vecTyNumElems != 2 && vecTyNumElems != 1)
      printf("Invalid vecTyNumElems %d\n", vecTyNumElems);
#endif
    for (uint reg_i = 0; reg_i < CRegRows; reg_i += vecTyNumElems) {
      if (vecTyNumElems > 1 && MaxColsA == MaxColsC && false) { // && !distParams.storeToDistMems
        //TODO: Cannot shA here if MaxColA < MaxColsC
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
            cCol = tile_k * (MaxColsC/kronCols) + 
                (cCol/(MaxColsC/kronCols)) * (colsC/kronCols) +
                cCol%(MaxColsC/kronCols);
          }
          if (TileSizeKronCols != MaxKronCols) {
            uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, TileSizeKronCols>();
            cCol += external_tile_kp_n*(colsC/(MaxKronCols/TileSizeKronCols)); 
          }

          const uint cIdx = cRow * colsC + cCol;
          if (cCol < colsC) {
            params.glC[cIdx] = shA[0][(tid/wSz)*wSz*vecTyNumElems + shVecI];
          }
        }
        __syncwarp();
      } else {
        const uint cRow = (rowA + tileRowA);
        uint cCol = outerTileKronCol*(MaxColsA/MaxKronRows) +
                    reg_j*(MaxColsA/MaxKronRows) +
                    tileColC + 
                    reg_i;
        // if (threadIdx.x == 0 and blockIdx.x == 0) printf("MaxColsC %d\n", MaxColsC);
        if (!K_EQUALS_VAR) {
          uint tile_k = get_tile_k<MaxKronCols, TileSizeKronCols>();
          cCol = tile_k * (MaxColsC/kronCols) + 
              (cCol/(MaxColsC/kronCols)) * (colsC/kronCols) +
              cCol%(MaxColsC/kronCols);
        }
        if (TileSizeKronCols != MaxKronCols) {
          uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, TileSizeKronCols>();
          cCol += external_tile_kp_n*(colsC/(MaxKronCols/TileSizeKronCols)); 
        }
        
        uint cIdx;
        ElemT* __restrict__ outputArray;

        if (distParams.storeToDistMems) {
          // kronRows = 64;
          // kronCols = 64;
          const uint perGPUK = (kronRows*kronRows*kronRows*kronRows)/2; //
          const uint numGPUs = 2; //distParams.numGPUs;

          const uint perGPUKByNumGPUs = distParams.perGPUKByNumGPUs;
          const uint perGPUKByKronRows = distParams.perGPUKByKronRows;
          const uint ColsAByKronRows = distParams.ColsAByKronRows;
          const uint gcMulUVAColsRatioKronRowsSquare = distParams.gcMulUVAColsRatioKronRowsSquare;

          const uint nextGc = cCol/perGPUKByNumGPUs;
          uint batchedKronMuls = distParams.LocalKrons;
          // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 0) printf("batchedKronMuls %d\n", batchedKronMuls);
          uint KronRowsPower = (batchedKronMuls == 3) ? kronRows*kronRows*kronRows : kronRows;//power(kronRows, batchedKronMuls);

          uint srcElem = cCol;
          uint UVAColsRatioKronRowsSquare = distParams.UVAColsRatioKronRowsSquare;// (perGPUK/KronRowsPower);
          uint withinP5 = gcMulUVAColsRatioKronRowsSquare +
                           ((srcElem%perGPUKByKronRows)/UVAColsRatioKronRowsSquare)*(distParams.ColsC/KronRowsPower) + //(perGPUK/UVAColsRatioKronRowsSquare)
                            srcElem % UVAColsRatioKronRowsSquare;
          uint p5Index = (srcElem/perGPUKByKronRows)*ColsAByKronRows;
          int newcCol = p5Index + withinP5;
          int gpuCol = newcCol - nextGc * perGPUK;
          cIdx = cRow * perGPUK + gpuCol;
          outputArray = (ElemT*)(distParams.getLocalGPUResult(nextGc));//(nextGc == 0) ? distParams.gpuResults1 : distParams.gpuResults2;
         
          // printf("outputArray %p\n", outputArray);
          // if (batchedKronMuls == 3 and regC[rowA][reg_i][reg_j] != )
          // if (threadIdx.x == 0) //((gpuCol >= perGPUK or gpuCol < 0)) 
          // printf("gpuCol %d nextGc %d perGPUK %d newcCol %d gc %d ColsA %d cIdx %d outputArray %p\n",
          //         gpuCol, nextGc, perGPUK, newcCol, distParams.gc, distParams.ColsA, cIdx, outputArray);
          // outputArray = distParams.gpuResults[nextGc];

          // if (threadIdx.x == 0 && blockIdx.x == 0 & blockIdx.y == 0) {
          //   uint nextGc = (distParams.gc == 0) ? 1 : 0;
          //   printf("Writing from %d to %d at %p\n", distParams.gc, nextGc, &distParams.gpuResults[nextGc]);
          //   distParams.gpuResults[nextGc][0] = 0;
          // }
        } else {
         cIdx = cRow * colsC + cCol;
         outputArray = params.glC;
        //  if (threadIdx.x == 0) printf("317: outputArray %p cIdx %d\n", outputArray, cIdx);
        }
        // assert(tid == cCol);
        // if (kp_idx == 0&& cRow == 0 && cCol < 64)
        //   printf("tid %d cCol %d outerTileKronCol %d tileColA %d reg_i %d reg_j %d\n", tid, cCol, outerTileKronCol, tileColA, reg_i, reg_j);
        //if (cCol < colsC) 
        
        {
          switch (vecTyNumElems) {
            case 4:
              globalStore4Elems(&outputArray[cIdx], 
                                regC[rowA][reg_i][reg_j], 
                                regC[rowA][reg_i+1][reg_j],
                                regC[rowA][reg_i+2][reg_j], 
                                regC[rowA][reg_i+3][reg_j]);
              break;
            case 2:
              globalStore2Elems(&outputArray[cIdx],
                                regC[rowA][reg_i][reg_j],
                                regC[rowA][reg_i+1][reg_j]);
              break;
            case 1: {
              globalStore1Elems(&outputArray[cIdx], regC[rowA][reg_i][reg_j]);
              // if (params.kp_idx == 2 && params.glC[cIdx] != 8.0f) {
              // if (params.kp_idx == 3 and blockIdx.y == 0 and cCol >= 4096) { //params.glC[cIdx] != 8.0f
              //   printf("kp_idx %d glC[%d] %f cRow %d cCol %d colsC %d MaxColsC %d tileColC %d outerTileKronCol %d\n",
              //          params.kp_idx, cIdx, params.glC[cIdx], cRow, cCol, colsC, MaxColsC, tileColC, outerTileKronCol, threadIdx.x);
              // }
              break;
            }
          }
        }
      }
    }}}
  }}
}