#include <stdio.h>
#include <type_traits>

#include "params.h"
#include "memory-transfer.cuh"

#pragma once

#define MIN(x,y)    (((x) < (y)) ? (x) : (y))
#define MAX(x,y)    (((x) > (y)) ? (x) : (y))
#define DIVUP(x, y) (((x) + (y) - 1)/((y)))

__host__ __device__ constexpr uint power(const uint x, const uint y) {
  uint result = 1;
  for (uint i = 0; i < y; i++) {
    result = result * x;
  }
  return result;
}

template<uint MaxKronCols, uint MaxTileSizeKronCols>
__device__ __forceinline__ uint get_tile_k() {return blockIdx.x/DIVUP(MaxKronCols, MaxTileSizeKronCols);}
template<uint MaxKronCols, uint MaxTileSizeKronCols>
__device__ __forceinline__ uint get_external_tile_kp_n() {return blockIdx.x%DIVUP(MaxKronCols, MaxTileSizeKronCols);}

__device__ __forceinline__ bool isfirstIdx(dim3 idx) {return idx.x == 0 && idx.y == 0 & idx.z == 0;}

template<typename ElemT>
__device__ __forceinline__
size_t nonAlignedElems(const ElemT* ptr, uint vecElems) {
  return (reinterpret_cast<size_t>(ptr)/sizeof(ElemT)) % vecElems;
}

template<typename ElemT, typename VecT, bool K_EQUALS_VAR, uint VecTNumElems>
__device__ __forceinline__ 
void shiftAgToAsh(const uint TileSizeColsA, const uint MaxKronRows,
                  const uint TileSizeKronRows, const uint MaxColsA,
                  const uint NumThreads, const uint CRegRows,
                  const uint kronRows, const uint colsA,
                  const uint tid, const uint tileKronRow, const uint rowA,
                  const uint a_col,
                  const uint tile_k, const uint external_tile_kp_k,
                  const ElemT* __restrict__ glRowAddr, ElemT* __restrict__ shA) {
  const ElemT* addrA;
  VecT  vec;
  ElemT elems[VecTNumElems];

  if (TileSizeKronRows == MaxKronRows) {
    addrA = &glRowAddr[(K_EQUALS_VAR ? 0 : tile_k*MaxColsA) + a_col];
  } else {
    addrA = &glRowAddr[(K_EQUALS_VAR ? 0 : tile_k*MaxColsA) + \
                  (a_col/TileSizeKronRows)*kronRows + external_tile_kp_k * TileSizeKronRows + tileKronRow + a_col % TileSizeKronRows];
  }

  globalLoadVec(addrA, vec);
  loadVecToRegs(vec, elems);

  #pragma unroll
  for (uint i = 0; i < VecTNumElems; i++) {
    uint ash_col = a_col + i;
    uint tileColA = (ash_col/TileSizeKronRows)/CRegRows;
    
    uint final_col = (ash_col/TileSizeKronRows)*TileSizeKronRows + 
                      (tileColA + ash_col%TileSizeKronRows)%TileSizeKronRows;
    shA[rowA * TileSizeColsA + final_col] = elems[i];
  }
}
 

template<typename ElemT, typename VecT, bool K_EQUALS_VAR>
__device__ __forceinline__ 
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

  for (uint rowA = 0; rowA < (RowsCModTileIsZero ? TileSizeRowsA : MIN(TileSizeRowsA, RowsC - tileRowA)); rowA += 1) {
    const ElemT* glRowAddr  = &glA[(rowA + tileRowA) * colsA];
    const size_t firstElems = 0; //nonAlignedElems(glRowAddr, VecTNumElems);
    const size_t lastElems  = 0; //TileSizeColsA % VecTNumElems;

    for (int a_col = tid; a_col < firstElems; a_col += NumThreads) {
      shiftAgToAsh<ElemT, ElemT, K_EQUALS_VAR, 1>(TileSizeColsA, MaxKronRows, TileSizeKronRows, MaxColsA, NumThreads, CRegRows, kronRows, colsA, tid, tileKronRow, rowA, a_col, tile_k, external_tile_kp_k, glRowAddr, shA);
    }

    for (int a_col = firstElems + tid*VecTNumElems; a_col < TileSizeColsA - lastElems; a_col += NumThreads*VecTNumElems) {
      shiftAgToAsh<ElemT, VecT, K_EQUALS_VAR, VecTNumElems>(TileSizeColsA, MaxKronRows, TileSizeKronRows, MaxColsA, NumThreads, CRegRows, kronRows, colsA, tid, tileKronRow, rowA, a_col, tile_k, external_tile_kp_k, glRowAddr, shA);
    }

    for (int a_col = TileSizeColsA - lastElems + tid; a_col < TileSizeColsA; a_col += NumThreads) {
      shiftAgToAsh<ElemT, ElemT, K_EQUALS_VAR, 1>(TileSizeColsA, MaxKronRows, TileSizeKronRows, MaxColsA, NumThreads, CRegRows, kronRows, colsA, tid, tileKronRow, rowA, a_col, tile_k, external_tile_kp_k, glRowAddr, shA);
    }
  }
}

template<typename ElemT, typename VecT>
__device__ __forceinline__ 
void tiledDirectFglToFsh(const uint MaxKronRows, const uint MaxKronCols, 
                         const uint TileSizeKronRows, const uint TileSizeKronCols,
                         const uint NumThreads, const uint external_tile_kp_n, const uint external_tile_kp_k, 
                         const uint tileKronRow, const uint kronRows, const uint kronCols, const uint tid, 
                         const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const int VecTNumElems = sizeof(VecT)/sizeof(ElemT);
  const uint loadInstr = MIN(TileSizeKronCols, VecTNumElems);
  //Create kronCols subwarps and each subwarp loads 0 to TileSizeKronRows elements
  const uint subWarps = MAX(1, NumThreads/(TileSizeKronCols/loadInstr));
  for (uint swid = tid/(TileSizeKronCols/loadInstr); swid < TileSizeKronRows; swid += subWarps) {
    ElemT elems[VecTNumElems];

    for (uint elem = tid%(TileSizeKronCols/loadInstr); elem < TileSizeKronCols/loadInstr; elem += NumThreads/subWarps) {
      const uint col = external_tile_kp_n*TileSizeKronCols + elem*loadInstr;
      const uint row = swid;

      const ElemT* addr = &Fgl[(external_tile_kp_k * TileSizeKronRows + tileKronRow + row) * kronCols + col];
      globalLoadVec_(addr, elems, loadInstr);
      
      #pragma unroll
      for (uint e = 0; e < loadInstr; e++) {
        uint linearIdx = elem*loadInstr + e;
        Fsh[row * TileSizeKronCols + linearIdx] = elems[e];
      }

      //This condition avoids generating the loop giving better performance
      if (TileSizeKronCols/loadInstr == NumThreads/subWarps) break;
    }
  }
}

template<typename ElemT, typename VecT>
__device__ __forceinline__ 
void fullDirectFglToFsh(const uint MaxKronRows, const uint MaxKronCols, 
                        const uint TileSizeKronRows, const uint TileSizeKronCols, 
                        const uint NumThreads, const uint kronRows, const uint kronCols, 
                        const uint tid, const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const int VecTNumElems = sizeof(VecT)/sizeof(ElemT);
  const uint loadInstr = MIN(kronRows*kronCols, VecTNumElems);
  const size_t sz = kronRows * kronCols;
  const int lastLoads = 0; //sz % loadInstr;

  for (uint eIdx = tid*loadInstr; eIdx < kronRows*kronCols - lastLoads; eIdx += blockDim.x*loadInstr) {
    ElemT regElems[VecTNumElems];
    VecT vec;

    vec = *(VecT*)&Fgl[eIdx];
    loadVecToRegs(vec, regElems);

    #pragma unroll
    for (uint vecElem = 0; vecElem < loadInstr; vecElem++) {
      uint idx = eIdx + vecElem;
      Fsh[(idx/MaxKronCols) * TileSizeKronCols + idx%MaxKronCols] = regElems[vecElem];
    }
  }

  for (uint eIdx = sz - lastLoads + tid; eIdx < sz; eIdx += blockDim.x) {
    ElemT regElem;
    regElem = Fgl[eIdx];
    Fsh[(eIdx/MaxKronCols) * TileSizeKronCols + eIdx%MaxKronCols] = regElem; 
  }
}