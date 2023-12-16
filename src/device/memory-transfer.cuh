#pragma once

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
__device__ __forceinline__ void globalLoadVec(const float* addr, float2& vec) {
  asm ("ld.ca.global.v2.f32 {%0, %1}, [%2];" : "=f"(vec.x), "=f"(vec.y) : "l"(addr));
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
__device__ __forceinline__ void loadVecToRegs(float2& vec, float* regs) {
  regs[0] = vec.x;
  regs[1] = vec.y;
}

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