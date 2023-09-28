#include <stdio.h>

#include "common.h"

#ifndef __DEVICE_FUNCTIONS__
#define __DEVICE_FUNCTIONS__

#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define DIVUP(x, y) (((x) + (y) - 1)/((y)))

#define CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess and e != cudaErrorPeerAccessAlreadyEnabled) {\
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

template<uint MaxKronCols, uint MaxTileSizeKronCols> __device__ uint get_tile_k() {return blockIdx.x/DIVUP(MaxKronCols, MaxTileSizeKronCols);}
template<uint MaxKronCols, uint MaxTileSizeKronCols> __device__ uint get_external_tile_kp_n() {return blockIdx.x%DIVUP(MaxKronCols, MaxTileSizeKronCols);}

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
template<uint x, uint y>
__device__ __host__ constexpr uint iconstpower() {
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
  // if (threadIdx.x == 0) printf("TileSizeRowsA %d\n", TileSizeRowsA);
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
        uint tileColA = (ash_col/TileSizeKronRows)/CRegRows; //(0,...,1024)/8 = (0,0,0,0,0,0 ... 127,127,127,127,127,127)
       
        uint final_col = (ash_col/TileSizeKronRows)*TileSizeKronRows + 
                         (tileColA + ash_col%TileSizeKronRows)%TileSizeKronRows;
        // if (blockIdx.x == 0&& blockIdx.y==0&&a_col < 64 && tileRowA == 0){
        //   printf("a_col %d final_col %d elem %f\n", a_col, rowA * TileSizeColsA +  final_col, elems[i]);
        // }
        // if (a_col +i < 8 && blockIdx.x == 0 && blockIdx.y == 0)
        //   printf("rowA %d a_col %d final_col %d TileSizeColsA %d elem %f\n", rowA, a_col + i, final_col, TileSizeColsA, elems[i]);
        shA[rowA * TileSizeColsA + final_col] = elems[i];
      }
    }
  }
}


template<typename ElemT, typename VecT, uint VecTNumElems>
__device__ __forceinline__ 
void tiledDirectFglToFsh(const uint MaxKronRows, const uint MaxKronCols, 
                         const uint TileSizeKronRows, const uint TileSizeKronCols,
                         const uint NumThreads, const uint external_tile_kp_n, const uint external_tile_kp_k, const uint tileKronRow, const uint kronRows, const uint kronCols, const uint tid, const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const uint loadInstr = MIN(TileSizeKronCols, VecTNumElems);
  //Create kronCols subwarps and each subwarp loads 0 to TileSizeKronRows elements
  for (uint swid = tid/(TileSizeKronCols/loadInstr); swid < TileSizeKronRows; swid += NumThreads/(TileSizeKronCols/loadInstr)) {
    VecT  vec;
    ElemT elems[VecTNumElems];

    const uint col = external_tile_kp_n*TileSizeKronCols + (tid%(TileSizeKronCols/loadInstr))*loadInstr;
    const uint row = swid;
    // shKronMats[tid%TileSizeKronRows][row] = glKronMats[(external_tile_kp_k * TileSizeKronCols + tileKronRow + row) * kronRows + col];

    globalLoadVec(&Fgl[(external_tile_kp_k * TileSizeKronRows + tileKronRow + row) * kronCols + col], vec);
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
void fullDirectFglToFsh(const uint MaxKronRows, const uint MaxKronCols, 
                        const uint TileSizeKronRows, const uint TileSizeKronCols, 
                        const uint NumThreads, const uint kronRows, const uint kronCols, 
                        const uint tid, const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const uint loadInstr = MIN(kronRows*kronCols, VecTNumElems);

  for (uint eIdx = tid*loadInstr; eIdx < kronRows*kronCols; eIdx += blockDim.x*loadInstr) {
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
}

#endif