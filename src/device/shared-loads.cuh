#include "kmm/matrix.h"
#include "device/register-loads.cuh"

template<typename ElemT, typename VecT>
CUDA_DEVICE
void storeAgToAsh(const uint TileM, const uint TileK, const uint MaxP,
                  const uint TileP, const uint MaxK,
                  const uint NumThreads, const uint CRegRows,
                  const uint P, const uint tid, const uint tileP, const uint tileM, const uint tileK,
                  const Matrix matrix, ElemT* __restrict__ Xsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);

  for (uint rowIdx = 0; rowIdx < (TileM == 1 ? TileM : MIN(TileM, matrix.m() - tileM)); rowIdx += 1) {
    const Matrix row = matrix.row<ElemT>(rowIdx + tileM);

    for (uint k = tid*VecTLen; k < TileK; k += NumThreads*VecTLen) {
      const ElemT* elemPtr;
      ElemT regs[VecTLen];

      if (TileP == MaxP)
        elemPtr = row.data<ElemT>(tileK*MaxK + k);
      else
        elemPtr = row.data<ElemT>(tileK*MaxK + (k/TileP)*P + tileP + k%TileP);

      ldGlobalVec((VecT*)(elemPtr), regs);
      
      #pragma unroll
      for (uint i = 0; i < VecTLen; i++) {
        //TODO: refactor based on paper
        uint shk = k + i;
        uint shTileK = (shk/TileP)/CRegRows;
        uint finalShK = (shk/TileP)*TileP + (shTileK + shk%TileP)%TileP;
        Xsh[rowIdx * TileK + finalShK] = regs[i];
      }
    }
  }
}

template<typename ElemT, typename VecT>
CUDA_DEVICE
void tiledDirectFglToFsh(const uint MaxP, const uint MaxKronCols, 
                         const uint TileP, const uint TileQ,
                         const uint NumThreads, const uint external_tile_kp_n,
                         const uint tileP, const uint P, const uint kronCols, const uint tid, 
                         const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  //Create kronCols subwarps and each subwarp loads 0 to TileP elements
  const uint subWarps = MAX(1, NumThreads/(TileQ/VecTLen));
  for (uint swid = tid/(TileQ/VecTLen); swid < TileP; swid += subWarps) {
    ElemT regs[VecTLen];

    for (uint elem = tid%(TileQ/VecTLen); elem < TileQ/VecTLen; elem += NumThreads/subWarps) {
      const uint col = external_tile_kp_n*TileQ + elem*VecTLen;
      const uint row = swid;

      VecT* addr = (VecT*)&Fgl[(tileP + row) * kronCols + col];
      ldGlobalVec(addr, regs);

      #pragma unroll
      for (uint e = 0; e < VecTLen; e++) {
        uint linearIdx = elem*VecTLen + e;
        Fsh[row * TileQ + linearIdx] = regs[e];
      }

      //This condition avoids generating the loop giving better performance
      if (TileQ/VecTLen == NumThreads/subWarps) break;
    }
  }
}

template<typename ElemT, typename VecT>
CUDA_DEVICE
void fullDirectFglToFsh(const uint MaxP, const uint MaxKronCols, 
                        const uint TileP, const uint TileQ,
                        const uint NumThreads, const uint P, const uint kronCols, 
                        const uint tid, const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  const size_t sz = P * kronCols;
  const int lastLoads = 0; //sz % loadInstr;

  //Use blockDim in loop adder instead of NumThreads because it generates better code 
  for (uint eIdx = tid*VecTLen; eIdx < P*kronCols - lastLoads; eIdx += blockDim.x*VecTLen) {
    ElemT regs[VecTLen];

    ldGlobalVec((VecT*)&Fgl[eIdx], regs);

    #pragma unroll
    for (uint ve = 0; ve < VecTLen; ve++) {
      uint idx = eIdx + ve;
      Fsh[(idx/MaxKronCols) * TileQ + idx%MaxKronCols] = regs[ve];
    }
  }

  // for (uint eIdx = sz - lastLoads + tid; eIdx < sz; eIdx += NumThreads) {
  //   ElemT regElem;
  //   regElem = Fgl[eIdx];
  //   Fsh[(eIdx/MaxKronCols) * TileQ + eIdx%MaxKronCols] = regElem; 
  // }
}