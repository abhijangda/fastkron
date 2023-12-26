#include "device/register-loads.cuh"

template<typename ElemT, typename VecT, uint VecTNumElems>
CUDA_DEVICE 
void shiftAgToAsh(const uint TileSizeColsA, const uint MaxKronRows,
                  const uint TileSizeKronRows, const uint MaxColsA,
                  const uint NumThreads, const uint CRegRows,
                  const uint kronRows, const uint colsA,
                  const uint tid, const uint tileKronRow, const uint rowA,
                  const uint a_col,
                  const uint tile_k, const uint external_tile_kp_k,
                  const ElemT* __restrict__ glRowAddr, ElemT* __restrict__ shA) {
  const ElemT* addrA;
  ElemT regs[VecTNumElems];

  if (TileSizeKronRows == MaxKronRows) {
    addrA = &glRowAddr[tile_k*MaxColsA + a_col];
  } else {
    addrA = &glRowAddr[tile_k*MaxColsA + \
                  (a_col/TileSizeKronRows)*kronRows + external_tile_kp_k * TileSizeKronRows + tileKronRow + a_col % TileSizeKronRows];
  }

  ldGlobalVec((VecT*)(addrA), regs);
  
  #pragma unroll
  for (uint i = 0; i < VecTNumElems; i++) {
    uint ash_col = a_col + i;
    uint tileColA = (ash_col/TileSizeKronRows)/CRegRows;
    
    uint final_col = (ash_col/TileSizeKronRows)*TileSizeKronRows + 
                      (tileColA + ash_col%TileSizeKronRows)%TileSizeKronRows;
    shA[rowA * TileSizeColsA + final_col] = regs[i];
  }
}
 

template<typename ElemT, typename VecT>
CUDA_DEVICE
void storeAgToAsh(const bool RowsCModTileIsZero, const uint TileSizeRowsA, 
                  const uint TileSizeColsA, const uint MaxKronRows,
                  const uint TileSizeKronRows, const uint MaxColsA,
                  const uint NumThreads, const uint CRegRows,
                  const uint RowsC, const uint kronRows, const uint colsA,
                  const uint tid, const uint tileKronRow, const uint tileRowA,
                  const uint tile_k, const uint external_tile_kp_k,
                  const ElemT* __restrict__ glA, ElemT* __restrict__ shA) {
  // if (threadIdx.x == 0) printf("TileSizeRowsA %d\n", TileSizeRowsA);
  const int VecTNumElems = sizeof(VecT)/sizeof(ElemT);

  for (uint rowA = 0; rowA < (TileSizeRowsA == 1 ? TileSizeRowsA : MIN(TileSizeRowsA, RowsC - tileRowA)); rowA += 1) {
    const ElemT* glRowAddr  = &glA[(rowA + tileRowA) * colsA];

    for (int a_col = tid*VecTNumElems; a_col < TileSizeColsA; a_col += NumThreads*VecTNumElems) {
      shiftAgToAsh<ElemT, VecT, VecTNumElems>(TileSizeColsA, MaxKronRows, TileSizeKronRows, MaxColsA, NumThreads, CRegRows, kronRows, colsA, tid, tileKronRow, rowA, a_col, tile_k, external_tile_kp_k, glRowAddr, shA);
    }
  }
}

template<typename ElemT, typename VecT>
CUDA_DEVICE
void tiledDirectFglToFsh(const uint MaxKronRows, const uint MaxKronCols, 
                         const uint TileSizeKronRows, const uint TileSizeKronCols,
                         const uint NumThreads, const uint external_tile_kp_n, const uint external_tile_kp_k, 
                         const uint tileKronRow, const uint kronRows, const uint kronCols, const uint tid, 
                         const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const int VecTNumElems = sizeof(VecT)/sizeof(ElemT);
  //Create kronCols subwarps and each subwarp loads 0 to TileSizeKronRows elements
  const uint subWarps = MAX(1, NumThreads/(TileSizeKronCols/VecTNumElems));
  for (uint swid = tid/(TileSizeKronCols/VecTNumElems); swid < TileSizeKronRows; swid += subWarps) {
    ElemT regs[VecTNumElems];

    for (uint elem = tid%(TileSizeKronCols/VecTNumElems); elem < TileSizeKronCols/VecTNumElems; elem += NumThreads/subWarps) {
      const uint col = external_tile_kp_n*TileSizeKronCols + elem*VecTNumElems;
      const uint row = swid;

      VecT* addr = (VecT*)&Fgl[(external_tile_kp_k * TileSizeKronRows + tileKronRow + row) * kronCols + col];
      ldGlobalVec(addr, regs);

      #pragma unroll
      for (uint e = 0; e < VecTNumElems; e++) {
        uint linearIdx = elem*VecTNumElems + e;
        Fsh[row * TileSizeKronCols + linearIdx] = regs[e];
      }

      //This condition avoids generating the loop giving better performance
      if (TileSizeKronCols/VecTNumElems == NumThreads/subWarps) break;
    }
  }
}

template<typename ElemT, typename VecT>
CUDA_DEVICE
void fullDirectFglToFsh(const uint MaxKronRows, const uint MaxKronCols, 
                        const uint TileSizeKronRows, const uint TileSizeKronCols, 
                        const uint NumThreads, const uint kronRows, const uint kronCols, 
                        const uint tid, const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const int VecTNumElems = sizeof(VecT)/sizeof(ElemT);
  const size_t sz = kronRows * kronCols;
  const int lastLoads = 0; //sz % loadInstr;

  for (uint eIdx = tid*VecTNumElems; eIdx < kronRows*kronCols - lastLoads; eIdx += blockDim.x*VecTNumElems) {
    ElemT regs[VecTNumElems];
    
    ldGlobalVec((VecT*)&Fgl[eIdx], regs);

    #pragma unroll
    for (uint vecElem = 0; vecElem < VecTNumElems; vecElem++) {
      uint idx = eIdx + vecElem;
      Fsh[(idx/MaxKronCols) * TileSizeKronCols + idx%MaxKronCols] = regs[vecElem];
    }
  }

  for (uint eIdx = sz - lastLoads + tid; eIdx < sz; eIdx += blockDim.x) {
    ElemT regElem;
    regElem = Fgl[eIdx];
    Fsh[(eIdx/MaxKronCols) * TileSizeKronCols + eIdx%MaxKronCols] = regElem; 
  }
}