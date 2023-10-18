#include <stdio.h>
#include <type_traits>

#include "params.h"

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

template<uint MaxKronCols, uint MaxTileSizeKronCols> __device__ __forceinline__ uint get_tile_k() {return blockIdx.x/DIVUP(MaxKronCols, MaxTileSizeKronCols);}
template<uint MaxKronCols, uint MaxTileSizeKronCols> __device__ __forceinline__ uint get_external_tile_kp_n() {return blockIdx.x%DIVUP(MaxKronCols, MaxTileSizeKronCols);}

__device__ __forceinline__ bool isfirstIdx(dim3 idx) {return idx.x == 0 && idx.y == 0 & idx.z == 0;}

__device__ __forceinline__ constexpr uint sqrt(uint x) {
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
__device__ __host__ __forceinline__ constexpr uint iconstpower() {
  uint result = 1;
  for (uint i = 0; i < y; i++) {
    result = result * x;
  }
  return result;
}

template<typename ElemT, typename Vec2T, typename Vec4T>
__device__ __forceinline__
void globalLoadVec2(const ElemT* addr, ElemT regs[], const uint vecSize) {
  switch(vecSize) {
    case 1: {
      regs[0] = *addr;
      break;
    }
    case 2: {
      Vec2T vec = *(Vec2T*)addr;
      regs[0] = vec.x; regs[1] = vec.y;
      break;
    }
    case 4: {
      Vec4T vec = *(Vec4T*)addr;
      regs[0] = vec.x; regs[1] = vec.y;
      regs[2] = vec.z; regs[3] = vec.w;
      break;
    }
  }
}

template<typename ElemT>
__device__ __forceinline__
void globalLoadVec_(const ElemT* __restrict__ addr, ElemT regs[], const uint vecSize) {
  // if (std::is_same<ElemT, float>::value) {
    globalLoadVec2<float, float2, float4>((float*)addr, regs, vecSize);
  // } else if (std::is_same<ElemT, int>::value) {
  //   // globalLoadVec2<int, int2, int4>((int*)addr, regs, vecSize);
  // } else {
  //   static_assert(std::is_same<ElemT, float>::value);
  // }
}

template<typename VecT, typename ElemT>
__device__ __forceinline__ void globalLoadVec(const ElemT* addr, VecT& vec) {
  //Not implemented
}

template<>
__device__ __forceinline__ void globalLoadVec(const float* addr, float4& vec) {
  asm ("ld.ca.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(vec.x), "=f"(vec.y), "=f"(vec.z), "=f"(vec.w) : "l"(addr));
}

template<>
__device__ __forceinline__ void globalLoadVec(const int* addr, int4& vec) {
  vec = *(int4*)addr;
}

template<>
__device__ __forceinline__ void globalLoadVec(const double* addr, double4& vec) {
  vec = *(double4*)addr;
}

template<>
__device__ __forceinline__ void globalLoadVec(const float* addr, float& vec) {
  vec = *addr;
}

template<typename VecT, typename ElemT>
__device__ __forceinline__ void loadVecToRegs(VecT& vec, ElemT* regs) {
  //Not implemented
}

//Four Element Vectors
template<typename VecT, typename ElemT>
__device__ __forceinline__ void load4ElemVecToRegs(VecT& vec, ElemT* regs) {
  regs[0] = vec.x;
  regs[1] = vec.y;
  regs[2] = vec.z;
  regs[3] = vec.w;
}

template<>
__device__ __forceinline__ void loadVecToRegs(float4& vec, float* regs) {
  load4ElemVecToRegs(vec, regs);
}

template<>
__device__ __forceinline__ void loadVecToRegs(int4& vec, int* regs) {
  load4ElemVecToRegs(vec, regs);
}


template<>
__device__ __forceinline__ void loadVecToRegs(double4& vec, double* regs) {
  load4ElemVecToRegs(vec, regs);
}

//Two element vectors
template<>
__device__ __forceinline__ void loadVecToRegs(double2& vec, double* regs) {
  regs[0] = vec.x;
  regs[1] = vec.y;
}


//Single element
template<>
__device__ __forceinline__ void loadVecToRegs(float& vec, float* regs) {
  regs[0] = vec;
}

//Store PTX instructions for each vector type
template<typename ElemT>
__device__ __forceinline__ void globalStore4Elems(ElemT* addr, ElemT elem1, ElemT elem2, ElemT elem3, ElemT elem4) {
}

template<>
__device__ __forceinline__ void globalStore4Elems(float* addr, float elem1, float elem2, float elem3, float elem4) {
  // asm ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "=f"(elem1), "=f"(elem2), "=f"(elem3), "=f"(elem4));
  float4 vec = {elem1, elem2, elem3, elem4};
  *(float4*)addr = vec;
}

template<>
__device__ __forceinline__ void globalStore4Elems(int* addr, int elem1, int elem2, int elem3, int elem4) {
  // asm ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "=f"(elem1), "=f"(elem2), "=f"(elem3), "=f"(elem4));
  int4 vec = {elem1, elem2, elem3, elem4};
  *(int4*)addr = vec;
}

template<>
__device__ __forceinline__ void globalStore4Elems(double* addr, double elem1, double elem2, double elem3, double elem4) {
  // asm ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "=f"(elem1), "=f"(elem2), "=f"(elem3), "=f"(elem4));
  double4 vec = {elem1, elem2, elem3, elem4};
  *(double4*)addr = vec;
}

template<typename ElemT>
__device__ __forceinline__ void globalStore2Elems(ElemT* addr, ElemT elem1, ElemT elem2) {
}

template<>
__device__ __forceinline__ void globalStore2Elems(float* addr, float elem1, float elem2) {
  float2 vec = {elem1, elem2};
  *(float2*)addr = vec;
}

template<>
__device__ __forceinline__ void globalStore2Elems(int* addr, int elem1, int elem2) {
  int2 vec = {elem1, elem2};
  *(int2*)addr = vec;
}

template<>
__device__ __forceinline__ void globalStore2Elems(double* addr, double elem1, double elem2) {
  double2 vec = {elem1, elem2};
  *(double2*)addr = vec;
}

template<typename ElemT>
__device__ __forceinline__ void globalStore1Elems(ElemT* addr, ElemT elem1) {
  *addr = elem1;
}

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


template<typename ElemT, typename VecT, bool K_EQUALS_VAR, uint VecTNumElems>
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
  for (uint rowA = 0; rowA < (RowsCModTileIsZero ? TileSizeRowsA : MIN(TileSizeRowsA, RowsC - tileRowA)); rowA += 1) {
    const ElemT* glRowAddr  = &glA[(rowA + tileRowA) * colsA];
    const size_t firstElems = 0; //nonAlignedElems(glRowAddr, VecTNumElems);
    const size_t lastElems  = 0; //TileSizeColsA % VecTNumElems;

    for (uint a_col = tid; a_col < firstElems; a_col += NumThreads) {
      shiftAgToAsh<ElemT, ElemT, K_EQUALS_VAR, 1>(TileSizeColsA, MaxKronRows, TileSizeKronRows, MaxColsA, NumThreads, CRegRows, kronRows, colsA, tid, tileKronRow, rowA, a_col, tile_k, external_tile_kp_k, glRowAddr, shA);
    }

    for (uint a_col = firstElems + tid*VecTNumElems; a_col < TileSizeColsA - lastElems; a_col += NumThreads*VecTNumElems) {
      shiftAgToAsh<ElemT, VecT, K_EQUALS_VAR, VecTNumElems>(TileSizeColsA, MaxKronRows, TileSizeKronRows, MaxColsA, NumThreads, CRegRows, kronRows, colsA, tid, tileKronRow, rowA, a_col, tile_k, external_tile_kp_k, glRowAddr, shA);
    }

    for (uint a_col = TileSizeColsA - lastElems + tid; a_col < TileSizeColsA; a_col += NumThreads) {
      shiftAgToAsh<ElemT, ElemT, K_EQUALS_VAR, 1>(TileSizeColsA, MaxKronRows, TileSizeKronRows, MaxColsA, NumThreads, CRegRows, kronRows, colsA, tid, tileKronRow, rowA, a_col, tile_k, external_tile_kp_k, glRowAddr, shA);
    }
  }
}

template<typename ElemT, typename VecT, uint VecTNumElems>
__device__ __forceinline__ 
void tiledDirectFglToFsh(const uint MaxKronRows, const uint MaxKronCols, 
                         const uint TileSizeKronRows, const uint TileSizeKronCols,
                         const uint NumThreads, const uint external_tile_kp_n, const uint external_tile_kp_k, 
                         const uint tileKronRow, const uint kronRows, const uint kronCols, const uint tid, 
                         const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const uint loadInstr = MIN(TileSizeKronCols, VecTNumElems);
  //Create kronCols subwarps and each subwarp loads 0 to TileSizeKronRows elements
  for (uint swid = tid/(TileSizeKronCols/loadInstr); swid < TileSizeKronRows; swid += NumThreads/(TileSizeKronCols/loadInstr)) {
    VecT  vec;
    ElemT elems[VecTNumElems];

    const uint col = external_tile_kp_n*TileSizeKronCols + (tid%(TileSizeKronCols/loadInstr))*loadInstr;
    const uint row = swid;
    // shKronMats[tid%TileSizeKronRows][row] = glKronMats[(external_tile_kp_k * TileSizeKronCols + tileKronRow + row) * kronRows + col];

    const ElemT* addr = &Fgl[(external_tile_kp_k * TileSizeKronRows + tileKronRow + row) * kronCols + col];
    globalLoadVec_(addr, elems, loadInstr);
    
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

#endif