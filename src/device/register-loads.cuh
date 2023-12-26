#include "config.h"

#pragma once

/***float loads***/
CUDA_DEVICE void ldGlobalVec(const float4* addr, float regs[4]) {
  float4 vec;
  asm ("ld.ca.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(vec.x), "=f"(vec.y), "=f"(vec.z), "=f"(vec.w) : "l"(addr));
  regs[0] = vec.x;
  regs[1] = vec.y;
  regs[2] = vec.z;
  regs[3] = vec.w;
}

CUDA_DEVICE void ldGlobalVec(const float2* addr, float regs[2]) {
  float2 vec;
  asm ("ld.ca.global.v2.f32 {%0, %1}, [%2];" : "=f"(vec.x), "=f"(vec.y) : "l"(addr));
  regs[0] = vec.x;
  regs[1] = vec.y;
}

CUDA_DEVICE void ldGlobalVec(const float* addr, float regs[1]) {
  asm ("ld.ca.global.f32 {%0}, [%1];" : "=f"(regs[0]) : "l"(addr));
}

//int loads
CUDA_DEVICE void ldGlobalVec(const int* addr, int4& vec) {
  vec = *(int4*)addr;
}

//double loads
CUDA_DEVICE void ldGlobalVec(const double* addr, double4& vec) {
  vec = *(double4*)addr;
}

//Store PTX instructions for each vector type
template<typename ElemT>
CUDA_DEVICE void globalStore4Elems(ElemT* addr, ElemT elem1, ElemT elem2, ElemT elem3, ElemT elem4) {
}

template<>
CUDA_DEVICE void globalStore4Elems(float* addr, float elem1, float elem2, float elem3, float elem4) {
  // asm ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "=f"(elem1), "=f"(elem2), "=f"(elem3), "=f"(elem4));
  float4 vec = {elem1, elem2, elem3, elem4};
  *(float4*)addr = vec;
}

template<>
CUDA_DEVICE void globalStore4Elems(int* addr, int elem1, int elem2, int elem3, int elem4) {
  // asm ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "=f"(elem1), "=f"(elem2), "=f"(elem3), "=f"(elem4));
  int4 vec = {elem1, elem2, elem3, elem4};
  *(int4*)addr = vec;
}

template<>
CUDA_DEVICE void globalStore4Elems(double* addr, double elem1, double elem2, double elem3, double elem4) {
  // asm ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "=f"(elem1), "=f"(elem2), "=f"(elem3), "=f"(elem4));
  double4 vec = {elem1, elem2, elem3, elem4};
  *(double4*)addr = vec;
}

template<typename ElemT>
CUDA_DEVICE void globalStore2Elems(ElemT* addr, ElemT elem1, ElemT elem2) {
}

template<>
CUDA_DEVICE void globalStore2Elems(float* addr, float elem1, float elem2) {
  float2 vec = {elem1, elem2};
  *(float2*)addr = vec;
}

template<>
CUDA_DEVICE void globalStore2Elems(int* addr, int elem1, int elem2) {
  int2 vec = {elem1, elem2};
  *(int2*)addr = vec;
}

template<>
CUDA_DEVICE void globalStore2Elems(double* addr, double elem1, double elem2) {
  double2 vec = {elem1, elem2};
  *(double2*)addr = vec;
}

template<typename ElemT>
CUDA_DEVICE void globalStore1Elems(ElemT* addr, ElemT elem1) {
  *addr = elem1;
}