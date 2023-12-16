#include "device_functions.cuh"

#include <type_traits>
#include <typeinfo>

template<typename ElemT, typename Vec2T, typename Vec4T,
         uint NumThreads, 
         uint MaxQ, uint MaxP, uint TileQ, uint TileK,
         uint TileM, uint NumFusedKerns, bool DistributeToGPUs, 
         uint CRegRows, uint CRegCols,
         uint KPK_EQUALS_VAR, uint TileP, 
         int AAlignment, int KronAlignment>
__launch_bounds__(NumThreads)
__global__ void kronGemmKernel(KernelParams<NumFusedKerns> params,
                               FusedParams<NumFusedKerns> fusedParams,
                               DistributedParams distParams,
                               EpilogueParams epilogueParams) {  
  static_assert(AAlignment == 1    || AAlignment == 2    || AAlignment == 4,
                "Alignment of A should be 1, 2 or 4");
  static_assert(KronAlignment == 1 || KronAlignment == 2 || KronAlignment == 4,
                "Alignment of A should be 1, 2 or 4");
  using XVecT = typename std::conditional<AAlignment == 1, ElemT, 
                  typename std::conditional<AAlignment == 2, Vec2T, 
                                            Vec4T>::type>::type;
  using FVecT = typename std::conditional<KronAlignment == 1, ElemT, 
                  typename std::conditional<KronAlignment == 2, Vec2T, 
                                            Vec4T>::type>::type;
  static_assert(0 < TileQ && TileQ <= MaxQ, "");
  static_assert(NumFusedKerns == 1 ||
                (NumFusedKerns > 1 && TileP >= MaxP && TileQ >= MaxQ),
                "Invalid tile size params for fusion");
  static_assert(TileK % MaxP          == 0, "TileK is not a multiple of MaxP");
  static_assert((TileK/MaxP)%CRegRows == 0, "CRegRows not a multiple of MaxCols/MaxP");

  const uint tid     = threadIdx.x;
  const uint ShTileP = MIN(TileP, MaxP);
  const uint ShTileK = TileK/(MaxP/ShTileP);

  register   ElemT regC[TileM][CRegRows][CRegCols] = {0};
  __shared__ ElemT Xsh[TileM][ShTileK];
  __shared__ ElemT Fsh[ShTileP][TileQ];

  uint Q;
  uint P;
  const uint K = params.k;
  const uint L = params.l;
 
  if (KPK_EQUALS_VAR) {
    Q = MaxQ;
    P = MaxP;
  } else {
    Q = params.qs[0];
    P = params.ps[0];
  }

  const ElemT* __restrict__ Xgl = (const ElemT*) params.x;

  const uint RegTileSizeACols = MIN(8, TileQ);
  
  const uint external_tile_kp_k = blockIdx.z;
  const uint external_tile_kp_n = get_external_tile_kp_n<MaxQ, TileQ>();
  const uint MaxL = (TileK/MaxP)*MaxQ;
  constexpr uint wSz = (TileK/MaxP)/CRegRows;

  const uint kp_col_start_ = (tid / wSz) * CRegCols;
  const uint a_col_start_  = (tid % wSz) * CRegRows;

  const uint tileRowA         = blockIdx.y * TileM;
  const uint outerTileKronCol = kp_col_start_;
  const uint tileColC         = a_col_start_ ;

  bool isThreadValid = (kp_col_start_ + CRegCols <= TileQ);
  uint tile_k = get_tile_k<MaxQ, TileQ>();
  
  for (uint tileKronRow = 0; tileKronRow < P; tileKronRow += ShTileP) {
    //Loop iterates only once when NumFusedKerns == 1
    storeAgToAsh<ElemT, XVecT, 0>(0, TileM, ShTileK, 
                                  MaxP, ShTileP, TileK, NumThreads, CRegRows, params.m,
                                  P, K, tid, 
                                  tileKronRow, tileRowA, 
                                  tile_k, external_tile_kp_k, 
                                  Xgl, &Xsh[0][0]);

    #pragma unroll
    for (int fusedFac = 0; fusedFac < NumFusedKerns; fusedFac++) {
      if (NumFusedKerns > 1) {
        #pragma unroll
        for (uint r = 0; r < TileM   ; r++) {
        #pragma unroll
        for (uint i = 0; i < CRegRows; i++) {
        #pragma unroll
        for (uint j = 0; j < CRegCols; j++) {
          regC[r][i][j] = 0;
        }}}
      }
      const ElemT* __restrict__ Fgl = (ElemT*)params.fs[fusedFac];
      if (TileQ == MaxQ && ShTileP == MaxP) {
        //Optimized to load full factor matrix
        fullDirectFglToFsh<ElemT, FVecT>(MaxP, MaxQ,
                                            ShTileP, TileQ, 
                                            NumThreads, P, 
                                            Q, tid, Fgl, &Fsh[0][0]);
      } else if (!(TileQ == MaxQ && ShTileP == MaxP)) {
        tiledDirectFglToFsh<ElemT, FVecT>(MaxP, MaxQ,
                                             ShTileP, TileQ, 
                                             NumThreads, external_tile_kp_n,
                                             external_tile_kp_k, tileKronRow, 
                                             P, Q, tid, Fgl,
                                             &Fsh[0][0]);
      } else {
      }
      __syncthreads();

      if (isThreadValid) {
      //Load RegTileSizeACols elements at a time to limit the register usage
      for (uint regTileACol = 0; regTileACol < ShTileP; regTileACol += RegTileSizeACols) {
        register ElemT Xr[TileM][CRegRows][RegTileSizeACols];
        register ElemT Fr[RegTileSizeACols][CRegCols];
        const uint tileColA = (tileColC);// * TileK)/MaxL;
        uint round_start = (tileColA / CRegRows)%ShTileP;

        #pragma unroll
        for (uint rowA = 0; rowA < TileM; rowA++) {
        if (TileM == 1 || rowA < params.m - tileRowA) {
          #pragma unroll
          for (uint rowC = 0; rowC < CRegRows; rowC++) {
            uint shACol = tileColA + rowC;
            #pragma unroll
            for (uint colC = 0; colC < RegTileSizeACols; colC++) {
              ElemT temp = Xsh[rowA][shACol * ShTileP + (regTileACol + colC + round_start)%ShTileP];
              Xr[rowA][rowC][colC] = temp;
            }
        }}}
        
        #pragma unroll
        for (uint colC = 0; colC < CRegCols; colC++) {
          uint shKronCol = outerTileKronCol + colC;
          #pragma unroll
          for (uint elem = 0; elem < RegTileSizeACols; elem++) {
            if (regTileACol + elem < ShTileP)
              Fr[elem][colC] = Fsh[regTileACol + elem][shKronCol];
          }
        }

        //Matrix Multiply Accumulate
        #pragma unroll
        for (uint rowA = 0; rowA < TileM; rowA++)
        if (TileM == 1 || rowA < params.m - tileRowA) {
          #pragma unroll
          for (uint i = 0;    i < CRegRows;         i++)
          #pragma unroll
          for (uint j = 0;    j < CRegCols;         j++)
          #pragma unroll
          for (uint k = 0;    k < RegTileSizeACols; k++) {
            if (k < ShTileP - regTileACol)
              regC[rowA][i][j] += Xr[rowA][i][k] * Fr[k][j];
          }
        }
      }
      }

      __syncthreads();
      if (isThreadValid && NumFusedKerns > 1 && fusedFac < NumFusedKerns - 1) {
      //Store C to shared memory using shift method
      for (int rowA = 0; rowA < TileM; rowA++) {
      if (TileM == 1 || rowA < params.m - tileRowA) {
        #pragma unroll
        for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
        for (uint reg_i = 0; reg_i < CRegRows; reg_i++) {
          uint cCol = outerTileKronCol*(TileK/MaxP) + reg_j*(TileK/MaxP) + tileColC + reg_i;
          uint tileColC = (cCol/ShTileP)/CRegRows;
          
          cCol = (cCol/ShTileP)*ShTileP + (tileColC + cCol%ShTileP)%ShTileP;
          Xsh[rowA][cCol] = regC[rowA][reg_i][reg_j];
      }}}}}
      __syncthreads();
    }
  }

  if (!isThreadValid) return;

  if (NumFusedKerns > 1) {
    for (uint rowShC = 0; rowShC < TileM; rowShC++) {
    if (TileM == 1 || rowShC < params.m - tileRowA) {
      //TODO: Improve below code like in the paper
      //TODO: Can be provided when compiling kernel.
      // #ifdef KPK_EQUALS_VAR
      // const uint KronRowsPower = iconstpower<P, NumFusedKerns>();
      // #else
      // const uint KronRowsPower = power(P, NumFusedKerns);
      // #endif
      uint UVAColsRatioKronColsSquare = fusedParams.UVAColsRatioKronColsSquare; //(ShTileK/KronRowsPower);
      const uint ColsCByKronColsPower = fusedParams.ColsCByKronColsPower; //L/KronRowsPower;//(L/(ShTileK/UVAColsRatioKronRowsSquare));
      const uint TileSizeColsAByKronCols = ShTileK/Q; //params.TileSizeColsAByKronRows;
      #pragma unroll
      for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
      for (uint reg_i = 0; reg_i < CRegRows; reg_i++) {
        uint colShC = outerTileKronCol*(TileK/MaxP) + reg_j*(TileK/MaxP) + tileColC + reg_i;
        const uint rowC = rowShC + tileRowA;
        uint withinP5 = tile_k * UVAColsRatioKronColsSquare +
                        ((colShC%TileSizeColsAByKronCols)/UVAColsRatioKronColsSquare)*ColsCByKronColsPower + 
                        colShC%UVAColsRatioKronColsSquare;
        
        uint p5Index = (colShC/TileSizeColsAByKronCols)*(L/Q);
        uint cCol = p5Index + withinP5;
        
        uint cIdx;
        ElemT* __restrict__ outputArray;

        if (DistributeToGPUs) {
          /// uint batchedKronMuls = distParams.LocalKrons;
          // uint KronRowsPower = (batchedKronMuls == 3) ? P*P*P : P;//power(P, batchedKronMuls);
          //TODO: Remove distParams. members that are not accessed here?
          uint UVAColsRatioKronRowsSquare = distParams.UVAColsRatioKronRowsSquare;//(perGPUK/KronRowsPower); //
          const uint perGPUNByNumGPUs = distParams.perGPUNByNumGPUs;
          const uint perGPUNByKronCols = distParams.perGPUNByKronCols;
          const uint ColsCByKronCols = distParams.ColsCByKronCols;
          const uint gcMulUVAColsRatioKronRowsSquare = distParams.gcMulUVAColsRatioKronRowsSquare;
          const uint ColsCByKronColsPower = distParams.ColsCByKronColsPower;
          
          const uint nextGc = cCol/perGPUNByNumGPUs;
          // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 0) printf("batchedKronMuls %d\n", batchedKronMuls);
          
          const uint perGPUN = L;
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
          cIdx = rowC * L + cCol;
          outputArray = (ElemT*)params.y;
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
  for (int rowA = 0; rowA < TileM; rowA++) {
  if (TileM == 1 || rowA < params.m - tileRowA) {
    #pragma unroll
    for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
    //Three least significant bits of CRegRows can be either 4, 2, or 1
    constexpr uint vecTyNumElems = MIN(AAlignment, MIN(CRegRows, 4) & (8 - 1));
    assert(vecTyNumElems == 4 || vecTyNumElems == 2 || vecTyNumElems == 1);
    for (uint reg_i = 0; reg_i < CRegRows; reg_i += vecTyNumElems) {
      const uint cRow = (rowA + tileRowA);
      uint cCol = outerTileKronCol*(TileK/MaxP) +
                  reg_j*(TileK/MaxP) +
                  tileColC +
                  reg_i;
      {
        cCol = tile_k * (MaxL/Q) +
                (cCol/(MaxL/Q)) * (L/Q) +
                cCol%(MaxL/Q);
      }
      if (TileQ != MaxQ) {
        uint external_tile_kp_n = get_external_tile_kp_n<MaxQ, TileQ>();
        cCol += external_tile_kp_n*(L/(MaxQ/TileQ)); 
      }

      uint cIdx;
      ElemT* __restrict__ outputArray;

      if (DistributeToGPUs) {
        // uint batchedKronMuls = distParams.LocalKrons;
        // uint KronRowsPower = (batchedKronMuls == 3) ? P*P*P : P;//power(P, batchedKronMuls);
        //TODO: Remove distParams. members that are not accessed here?
        uint UVAColsRatioKronRowsSquare = distParams.UVAColsRatioKronRowsSquare;//(perGPUK/KronRowsPower); //
        const uint perGPUNByNumGPUs = distParams.perGPUNByNumGPUs;
        const uint perGPUNByKronCols = distParams.perGPUNByKronCols;
        const uint ColsCByKronCols = distParams.ColsCByKronCols;
        const uint gcMulUVAColsRatioKronRowsSquare = distParams.gcMulUVAColsRatioKronRowsSquare;
        const uint ColsCByKronColsPower = distParams.ColsCByKronColsPower;
        
        uint nextGc = cCol/perGPUNByNumGPUs;
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 0) printf("batchedKronMuls %d\n", batchedKronMuls);
        
        const uint perGPUN = L;
        uint srcElem = cCol;
        uint withinP5 = gcMulUVAColsRatioKronRowsSquare +
                        ((srcElem%perGPUNByKronCols)/UVAColsRatioKronRowsSquare)*ColsCByKronColsPower + //(perGPUK/UVAColsRatioKronRowsSquare)
                        srcElem % UVAColsRatioKronRowsSquare;
        uint p5Index = (srcElem/perGPUNByKronCols)*ColsCByKronCols;
        int newcCol = p5Index + withinP5;
        int gpuCol = newcCol - nextGc * perGPUN;
        cIdx = cRow * perGPUN + gpuCol;
        outputArray = (ElemT*)(distParams.getLocalGPUResult(nextGc));//(nextGc == 0) ? distParams.gpuResults1 : distParams.gpuResults2;

        // if (params.kp_idx == 0 && blockIdx.y == 0) {//(gpuCol >= perGPUN || gpuCol < 0) {
        //   printf("344 outputArray %p nextGc %d cIdx %d perGPUN %d\n", outputArray, nextGc, cIdx, perGPUN);
        // }
        
        // printf("outputArray %p\n", outputArray);
        // if (batchedKronMuls == 3 and regC[rowA][reg_i][reg_j] != )
        // if (threadIdx.x == 0) //((gpuCol >= perGPUK or gpuCol < 0)) 
        // printf("gpuCol %d nextGc %d perGPUK %d newcCol %d gc %d K %d cIdx %d outputArray %p\n",
        //         gpuCol, nextGc, perGPUK, newcCol, distParams.gc, distParams.K, cIdx, outputArray);
        // outputArray = distParams.gpuResults[nextGc];

        // if (threadIdx.x == 0 && blockIdx.x == 0 & blockIdx.y == 0) {
        //   uint nextGc = (distParams.gc == 0) ? 1 : 0;
        //   printf("Writing from %d to %d at %p\n", distParams.gc, nextGc, &distParams.gpuResults[nextGc]);
        //   distParams.gpuResults[nextGc][0] = 0;
        // }
      } else {
        cIdx = cRow * L + cCol;
        outputArray = (ElemT*)params.y;
      //  if (threadIdx.x == 0) printf("317: outputArray %p cIdx %d\n", outputArray, cIdx);
      }

      if (params.kp_idx == 0) {
        for (int i = 0; i < vecTyNumElems; i++) {
          ElemT d = epilogueParams.getBeta<ElemT>() * ((epilogueParams.getD<ElemT>() != nullptr) ? epilogueParams.getD<ElemT>()[cIdx + i] : 0);
          regC[rowA][reg_i + i][reg_j] = epilogueParams.getAlpha<ElemT>() * regC[rowA][reg_i + i][reg_j] + d;
        }
      }
      
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
          //   printf("kp_idx %d glC[%d] %f cRow %d cCol %d L %d MaxL %d tileColC %d outerTileKronCol %d\n",
          //          params.kp_idx, cIdx, params.glC[cIdx], cRow, cCol, L, MaxL, tileColC, outerTileKronCol, threadIdx.x);
          // }
          break;
        }
      }
    }}}
  }}
}