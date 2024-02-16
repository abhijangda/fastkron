#include <stdio.h>
#include <type_traits>

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

__device__ __forceinline__ bool isfirstIdx(dim3 idx) {return idx.x == 0 && idx.y == 0 & idx.z == 0;}

template<typename ElemT>
__device__ __forceinline__
size_t nonAlignedElems(const ElemT* ptr, uint vecElems) {
  return (reinterpret_cast<size_t>(ptr)/sizeof(ElemT)) % vecElems;
}