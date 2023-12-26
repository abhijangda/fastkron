#include "device/register-loads.cuh"

template<typename ElemT, typename VecT, uint VecTLen>
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
  ElemT regs[VecTLen];

  if (TileSizeKronRows == MaxKronRows) {
    addrA = &glRowAddr[tile_k*MaxColsA + a_col];
  } else {
    addrA = &glRowAddr[tile_k*MaxColsA + \
                  (a_col/TileSizeKronRows)*kronRows + external_tile_kp_k * TileSizeKronRows + tileKronRow + a_col % TileSizeKronRows];
  }

  ldGlobalVec((VecT*)(addrA), regs);
  
  #pragma unroll
  for (uint i = 0; i < VecTLen; i++) {
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
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);

  for (uint rowA = 0; rowA < (TileSizeRowsA == 1 ? TileSizeRowsA : MIN(TileSizeRowsA, RowsC - tileRowA)); rowA += 1) {
    const ElemT* glRowAddr  = &glA[(rowA + tileRowA) * colsA];

    for (int a_col = tid*VecTLen; a_col < TileSizeColsA; a_col += NumThreads*VecTLen) {
      shiftAgToAsh<ElemT, VecT, VecTLen>(TileSizeColsA, MaxKronRows, TileSizeKronRows, MaxColsA, NumThreads, CRegRows, kronRows, colsA, tid, tileKronRow, rowA, a_col, tile_k, external_tile_kp_k, glRowAddr, shA);
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
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  //Create kronCols subwarps and each subwarp loads 0 to TileSizeKronRows elements
  const uint subWarps = MAX(1, NumThreads/(TileSizeKronCols/VecTLen));
  for (uint swid = tid/(TileSizeKronCols/VecTLen); swid < TileSizeKronRows; swid += subWarps) {
    ElemT regs[VecTLen];

    for (uint elem = tid%(TileSizeKronCols/VecTLen); elem < TileSizeKronCols/VecTLen; elem += NumThreads/subWarps) {
      const uint col = external_tile_kp_n*TileSizeKronCols + elem*VecTLen;
      const uint row = swid;

      VecT* addr = (VecT*)&Fgl[(external_tile_kp_k * TileSizeKronRows + tileKronRow + row) * kronCols + col];
      ldGlobalVec(addr, regs);

      #pragma unroll
      for (uint e = 0; e < VecTLen; e++) {
        uint linearIdx = elem*VecTLen + e;
        Fsh[row * TileSizeKronCols + linearIdx] = regs[e];
      }

      //This condition avoids generating the loop giving better performance
      if (TileSizeKronCols/VecTLen == NumThreads/subWarps) break;
    }
  }
}

template<typename ElemT, typename VecT>
CUDA_DEVICE
void fullDirectFglToFsh(const uint MaxKronRows, const uint MaxKronCols, 
                        const uint TileSizeKronRows, const uint TileSizeKronCols,
                        const uint NumThreads, const uint kronRows, const uint kronCols, 
                        const uint tid, const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  const size_t sz = kronRows * kronCols;
  const int lastLoads = 0; //sz % loadInstr;

  for (uint eIdx = tid*VecTLen; eIdx < kronRows*kronCols - lastLoads; eIdx += NumThreads*VecTLen) {
    ElemT regs[VecTLen];

    ldGlobalVec((VecT*)&Fgl[eIdx], regs);

    #pragma unroll
    for (uint ve = 0; ve < VecTLen; ve++) {
      uint idx = eIdx + ve;
      Fsh[(idx/MaxKronCols) * TileSizeKronCols + idx%MaxKronCols] = regs[ve];
    }
  }

  for (uint eIdx = sz - lastLoads + tid; eIdx < sz; eIdx += NumThreads) {
    ElemT regElem;
    regElem = Fgl[eIdx];
    Fsh[(eIdx/MaxKronCols) * TileSizeKronCols + eIdx%MaxKronCols] = regElem; 
  }
}