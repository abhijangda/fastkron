#include "device_functions.cuh"

#include <type_traits>
#include <typeinfo>

enum RowParallelismTy {
  Low = 0,
  Medium,
  High,
  Num = 3,
};

template<typename ElemT, typename Vec2T, typename Vec4T, uint NumThreads, RowParallelismTy RowParallelism, 
         uint TileSizeRowsA, bool RowsCModTileIsZero, uint MaxColsA, uint MaxKronCols, 
         uint MaxKronRows, uint TileSizeKronCols, uint K_EQUALS_VAR,
         uint KPK_EQUALS_VAR, uint CRegRows, uint CRegCols, uint SharedTileKronRows, uint NumFusedKerns, bool DistributeToGPUs,
         int AAlignment, int KronAlignment>
__launch_bounds__(NumThreads)
__global__ void kronGemmKernel(KernelParams<NumFusedKerns> params,
                               FusedParams<NumFusedKerns> fusedParams,
                               DistributedParams distParams,
                               EpilogueParams epilogueParams) {
  
  const uint WarpSize     = 32;
  const uint tid          = threadIdx.x;
  // const uint wid          = tid/WarpSize;
  // const uint lane         = tid%WarpSize;
  // const uint blockWarps   = blockDim.x/WarpSize;
  static_assert(AAlignment == 1 || AAlignment == 2 || AAlignment == 4, "Alignment of A should be 1, 2 or 4");
  static_assert(KronAlignment == 1 || KronAlignment == 2 || KronAlignment == 4, "Alignment of A should be 1, 2 or 4");
  using AVecT = typename std::conditional<AAlignment == 1, ElemT, typename std::conditional<AAlignment == 2, Vec2T, Vec4T>::type>::type;
  using KronVecT = typename std::conditional<KronAlignment == 1, ElemT, typename std::conditional<KronAlignment == 2, Vec2T, Vec4T>::type>::type;

  const uint TileSizeKronRows    = MIN(SharedTileKronRows,  MaxKronRows);
  const uint TileSizeColsA       = MaxColsA/(MaxKronRows/TileSizeKronRows);
  
  static_assert(0 < TileSizeKronCols && TileSizeKronCols <= MaxKronCols, "");
  static_assert(NumFusedKerns == 1 || 
                (NumFusedKerns > 1 && TileSizeKronRows >= MaxKronRows && TileSizeKronCols >= MaxKronCols),
                "Invalid tile size params for fusion");
  static_assert(MaxColsA % MaxKronRows == 0, "MaxColsA is not a multiple of MaxKronRows");
  static_assert((MaxColsA/MaxKronRows)%CRegRows == 0, "CRegRows not a multiple of MaxCols/MaxKronRows");

  register   ElemT regC[TileSizeRowsA][CRegRows][CRegCols] = {0};
  __shared__ ElemT shA[TileSizeRowsA][TileSizeColsA];
  __shared__ ElemT shKronMats[TileSizeKronRows][TileSizeKronCols];

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
  const ElemT* __restrict__ glA = (const ElemT*) params.glA;

  const uint RegTileSizeACols = MIN(8, TileSizeKronCols);
  
  const uint external_tile_kp_k = blockIdx.z;
  const uint external_tile_kp_n = get_external_tile_kp_n<MaxKronCols, TileSizeKronCols>();
  const uint MaxColsC = (MaxColsA/MaxKronRows)*MaxKronCols;
  constexpr uint wSz = (MaxColsA/MaxKronRows)/CRegRows; //31/4
  // constexpr uint wSz2 = (MaxKronCols/CRegCols); //

  const uint kp_col_start_ = (tid / wSz) * CRegCols;
  const uint a_col_start_  = (tid % wSz) * CRegRows;

  const uint tileRowA         = blockIdx.y * TileSizeRowsA;
  const uint outerTileKronCol = kp_col_start_;
  const uint tileColC         = a_col_start_ ;

  bool isThreadValid = (kp_col_start_ + CRegCols <= TileSizeKronCols);
  
  for (uint tileKronRow = 0; tileKronRow < kronRows; tileKronRow += TileSizeKronRows) {
    //Loop iterates only once when NumFusedKerns == 1
    uint tile_k = get_tile_k<MaxKronCols, TileSizeKronCols>();
    storeAgToAsh<ElemT, AVecT, K_EQUALS_VAR>(RowsCModTileIsZero, TileSizeRowsA, TileSizeColsA, 
                                            MaxKronRows, TileSizeKronRows, MaxColsA, NumThreads, CRegRows, params.RowsC,
                                            kronRows, colsA, tid, 
                                            tileKronRow, tileRowA, 
                                            tile_k, external_tile_kp_k, 
                                            glA, &shA[0][0]);

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
      const ElemT* __restrict__ glKronMat = (ElemT*)params.glKronMats[fusedFac];
      if (TileSizeKronCols == MaxKronCols && TileSizeKronRows == MaxKronRows) {
        //Optimized to load full factor matrix
        fullDirectFglToFsh<ElemT, KronVecT>(MaxKronRows, MaxKronCols,
                                            TileSizeKronRows, TileSizeKronCols, 
                                            NumThreads, kronRows, 
                                            kronCols, tid, glKronMat, &shKronMats[0][0]);
      } else if (!(TileSizeKronCols == MaxKronCols && TileSizeKronRows == MaxKronRows)) {
        tiledDirectFglToFsh<ElemT, KronVecT>(MaxKronRows, MaxKronCols,
                                             TileSizeKronRows, TileSizeKronCols, 
                                             NumThreads, external_tile_kp_n,
                                             external_tile_kp_k, tileKronRow, 
                                             kronRows, kronCols, tid, glKronMat,
                                             &shKronMats[0][0]);
      } else {
      }
      __syncthreads();

      if (isThreadValid) {
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
              {
                ElemT temp = shA[rowA][shACol * TileSizeKronRows + (regTileACol + colC + round_start)%TileSizeKronRows];
                Ar[rowA][rowC][colC] = temp;
              }
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
          for (uint elem = 0; elem < RegTileSizeACols; elem++) {
            if (regTileACol + elem < TileSizeKronRows)
              KPr[elem][colC] = shKronMats[regTileACol + elem][shKronCol];
          }
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
            if (k < TileSizeKronRows - regTileACol)
              regC[rowA][i][j] += Ar[rowA][i][k] * KPr[k][j];
            // if (Ar[rowA][i][k] == 0.0f) printf("Ar[rowA][%d][%d] %f\n", i,k, Ar[rowA][i][k]);
            // if (params.kp_idx == 1 and tileColC == 0 and outerTileKronCol == 0 and tileRowA == 0) {
            //   printf("regC %f KPr[k][j] %f Ar[rowA][i][k] %f\n", regC[rowA][i][j], KPr[k][j], Ar[rowA][i][k]);
            // }
          }
        }
      }
      }

      __syncthreads();
      if (isThreadValid && NumFusedKerns > 1 && fusedFac < NumFusedKerns - 1) {
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
      }}}}}
      __syncthreads();
    }
  }

  if (!isThreadValid) return;

  if (NumFusedKerns > 1) {
    for (uint rowShC = 0; rowShC < TileSizeRowsA; rowShC++) {
    if (RowsCModTileIsZero || (TileSizeRowsA > 1 && rowShC < params.RowsC - tileRowA)) {
      //TODO: Improve below code like in the paper
      //TODO: Can be provided when compiling kernel.
      // #ifdef KPK_EQUALS_VAR
      // const uint KronRowsPower = iconstpower<kronRows, NumFusedKerns>();
      // #else
      // const uint KronRowsPower = power(kronRows, NumFusedKerns);
      // #endif
      uint UVAColsRatioKronColsSquare = fusedParams.UVAColsRatioKronColsSquare; //(TileSizeColsA/KronRowsPower);
      const uint ColsCByKronColsPower = fusedParams.ColsCByKronColsPower; //colsC/KronRowsPower;//(colsC/(TileSizeColsA/UVAColsRatioKronRowsSquare));
      const uint TileSizeColsAByKronCols = TileSizeColsA/kronCols; //params.TileSizeColsAByKronRows;
      #pragma unroll
      for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
      for (uint reg_i = 0; reg_i < CRegRows; reg_i++) {
        uint colShC = outerTileKronCol*(MaxColsA/MaxKronRows) + reg_j*(MaxColsA/MaxKronRows) + tileColC + reg_i;
        const uint rowC = rowShC + tileRowA;
        uint tile_k = 0;
        if (!K_EQUALS_VAR) {
          tile_k = get_tile_k<MaxKronCols, TileSizeKronCols>();
        }
        uint withinP5 = tile_k * UVAColsRatioKronColsSquare +
                        ((colShC%TileSizeColsAByKronCols)/UVAColsRatioKronColsSquare)*ColsCByKronColsPower + 
                        colShC%UVAColsRatioKronColsSquare;
        
        uint p5Index = (colShC/TileSizeColsAByKronCols)*(colsC/kronCols);
        uint cCol = p5Index + withinP5;
        
        uint cIdx;
        ElemT* __restrict__ outputArray;

        if (DistributeToGPUs) {
          /// uint batchedKronMuls = distParams.LocalKrons;
          // uint KronRowsPower = (batchedKronMuls == 3) ? kronRows*kronRows*kronRows : kronRows;//power(kronRows, batchedKronMuls);
          //TODO: Remove distParams. members that are not accessed here?
          uint UVAColsRatioKronRowsSquare = distParams.UVAColsRatioKronRowsSquare;//(perGPUK/KronRowsPower); //
          const uint perGPUNByNumGPUs = distParams.perGPUNByNumGPUs;
          const uint perGPUNByKronCols = distParams.perGPUNByKronCols;
          const uint ColsCByKronCols = distParams.ColsCByKronCols;
          const uint gcMulUVAColsRatioKronRowsSquare = distParams.gcMulUVAColsRatioKronRowsSquare;
          const uint ColsCByKronColsPower = distParams.ColsCByKronColsPower;
          
          const uint nextGc = cCol/perGPUNByNumGPUs;
          // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 0) printf("batchedKronMuls %d\n", batchedKronMuls);
          
          const uint perGPUN = colsC;
          uint srcElem = cCol;
          uint withinP5 = gcMulUVAColsRatioKronRowsSquare +
                          ((srcElem%perGPUNByKronCols)/UVAColsRatioKronRowsSquare)*ColsCByKronColsPower + //(perGPUK/UVAColsRatioKronRowsSquare)
                          srcElem % UVAColsRatioKronRowsSquare;
          uint p5Index = (srcElem/perGPUNByKronCols)*ColsCByKronCols;
          int newcCol = p5Index + withinP5;
          int gpuCol = newcCol - nextGc * perGPUN;
          cIdx = rowC * perGPUN + gpuCol;
          outputArray = (ElemT*)(distParams.getLocalGPUResult(nextGc));
        } else {
          cIdx = rowC * colsC + cCol;
          outputArray = (ElemT* __restrict__)params.glC;
        }

        if (params.kp_idx == 0) {
          ElemT d = (epilogueParams.getD<ElemT>()) ? epilogueParams.getBeta<ElemT>() * epilogueParams.getD<ElemT>()[cIdx] : 0;
          outputArray[cIdx] = epilogueParams.getAlpha<ElemT>() * regC[rowShC][reg_i][reg_j] + d;
        } else {
          outputArray[cIdx] = regC[rowShC][reg_i][reg_j];
        }
    }}}
  }} else {
  #pragma unroll
  for (int rowA = 0; rowA < TileSizeRowsA; rowA++) {
  if (RowsCModTileIsZero || (TileSizeRowsA > 1 && rowA < params.RowsC - tileRowA)) {
    #pragma unroll
    for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
    //Three least significant bits of CRegRows can be either 4, 2, or 1
    constexpr uint vecTyNumElems = MIN(AAlignment, MIN(CRegRows, 4) & (8 - 1));
#ifndef EVAL
    if (vecTyNumElems != 4 && vecTyNumElems != 2 && vecTyNumElems != 1)
      printf("Invalid vecTyNumElems %d\n", vecTyNumElems);
#endif
    for (uint reg_i = 0; reg_i < CRegRows; reg_i += vecTyNumElems) {
      if (vecTyNumElems > 1 && MaxColsA == MaxColsC && false && !DistributeToGPUs) { // && !distParams.storeToDistMems
        //TODO: disabling this part because it is buggy and does not help much in performance
        //TODO: Cannot shA here if MaxColA < MaxColsC
        shA[0][tid * vecTyNumElems] = regC[rowA][reg_i][reg_j];
        shA[0][tid * vecTyNumElems+1] = regC[rowA][reg_i+1][reg_j];
        if (vecTyNumElems > 2) {
          shA[0][tid * vecTyNumElems+2] = regC[rowA][reg_i+2][reg_j];
          shA[0][tid * vecTyNumElems+3] = regC[rowA][reg_i+3][reg_j];
        }
        
        __syncthreads();
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
            ((ElemT* __restrict__)params.glC)[cIdx] = shA[0][(tid/wSz)*wSz*vecTyNumElems + shVecI];
          }
        }
        __syncwarp();
      } else {
        const uint cRow = (rowA + tileRowA);
        uint cCol = outerTileKronCol*(MaxColsA/MaxKronRows) + //0
                    reg_j*(MaxColsA/MaxKronRows) +//0-30*(496/16)=0,4,8,16,20,24,28,...,120
                    tileColC + //0 or 2
                    reg_i;//0 to 2
        uint cCol2 = cCol;
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

        if (DistributeToGPUs) {
          // uint batchedKronMuls = distParams.LocalKrons;
          // uint KronRowsPower = (batchedKronMuls == 3) ? kronRows*kronRows*kronRows : kronRows;//power(kronRows, batchedKronMuls);
          //TODO: Remove distParams. members that are not accessed here?
          uint UVAColsRatioKronRowsSquare = distParams.UVAColsRatioKronRowsSquare;//(perGPUK/KronRowsPower); //
          const uint perGPUNByNumGPUs = distParams.perGPUNByNumGPUs;
          const uint perGPUNByKronCols = distParams.perGPUNByKronCols;
          const uint ColsCByKronCols = distParams.ColsCByKronCols;
          const uint gcMulUVAColsRatioKronRowsSquare = distParams.gcMulUVAColsRatioKronRowsSquare;
          const uint ColsCByKronColsPower = distParams.ColsCByKronColsPower;
          
          uint nextGc = cCol/perGPUNByNumGPUs;
          // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 0) printf("batchedKronMuls %d\n", batchedKronMuls);
          
          const uint perGPUN = colsC;
          uint srcElem = cCol;
          uint withinP5 = gcMulUVAColsRatioKronRowsSquare +
                          ((srcElem%perGPUNByKronCols)/UVAColsRatioKronRowsSquare)*ColsCByKronColsPower + //(perGPUK/UVAColsRatioKronRowsSquare)
                          srcElem % UVAColsRatioKronRowsSquare;
          uint p5Index = (srcElem/perGPUNByKronCols)*ColsCByKronCols;
          int newcCol = p5Index + withinP5;
          int gpuCol = newcCol - nextGc * perGPUN;
          cIdx = cRow * perGPUN + gpuCol;
          outputArray = (ElemT* __restrict__)(distParams.getLocalGPUResult(nextGc));//(nextGc == 0) ? distParams.gpuResults1 : distParams.gpuResults2;
          // if (params.kp_idx == 0 && blockIdx.y == 0) {//(gpuCol >= perGPUN || gpuCol < 0) {
          //   printf("344 outputArray %p nextGc %d cIdx %d perGPUN %d\n", outputArray, nextGc, cIdx, perGPUN);
          // }
          
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
        //  if (threadIdx.x == 0)
        //   printf("363 cCol %d\n", cCol);
         outputArray = (ElemT* __restrict__)params.glC;
        //  if (threadIdx.x == 0) printf("317: outputArray %p cIdx %d\n", outputArray, cIdx);
        }
        // assert(tid == cCol);
        // if (kp_idx == 0&& cRow == 0 && cCol < 64)
        //   printf("tid %d cCol %d outerTileKronCol %d tileColA %d reg_i %d reg_j %d\n", tid, cCol, outerTileKronCol, tileColA, reg_i, reg_j);
        //if (cCol < colsC) 
        
        if (params.kp_idx == 0) {
          for (int i = 0; i < vecTyNumElems; i++) {
            ElemT d = epilogueParams.getBeta<ElemT>() * ((epilogueParams.getD<ElemT>() != nullptr) ? epilogueParams.getD<ElemT>()[cIdx + i] : 0);
            regC[rowA][reg_i + i][reg_j] = epilogueParams.getAlpha<ElemT>() * regC[rowA][reg_i + i][reg_j] + d;
          }
        }
        {
          switch (vecTyNumElems) {
            case 4: {
              globalStore4Elems(&outputArray[cIdx], 
                                regC[rowA][reg_i][reg_j], 
                                regC[rowA][reg_i+1][reg_j],
                                regC[rowA][reg_i+2][reg_j], 
                                regC[rowA][reg_i+3][reg_j]);
              break;
            }
            case 2: {
              globalStore2Elems(&outputArray[cIdx],
                                regC[rowA][reg_i][reg_j],
                                regC[rowA][reg_i+1][reg_j]);
              break;
            }
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