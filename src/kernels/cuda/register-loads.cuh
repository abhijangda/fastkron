#include "config.h"

#pragma once

/***float loads***/
CUDA_DEVICE 
void ldGlobalVec(const float4* ptr, float regs[4]) {
  #if defined(__NVCC__) || defined(__CUDACC__)
    asm volatile ("ld.ca.global.v4.f32 {%0, %1, %2, %3}, [%4];" :
                  "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]) : "l"(ptr));
  #elif defined(__HIPCC__)
    float4 f4 = *(float4*)ptr;
    regs[0] = f4.x; regs[1] = f4.y; regs[2] = f4.z; regs[3] = f4.w;
  #endif
}

CUDA_DEVICE
void ldGlobalVec(const float2* ptr, float regs[2]) {
  #if defined(__NVCC__) || defined(__CUDACC__)
    asm volatile ("ld.ca.global.v2.f32 {%0, %1}, [%2];" :
                  "=f"(regs[0]), "=f"(regs[1]) : "l"(ptr));
  #elif defined(__HIPCC__)
    float2 f2 = *(float2*)ptr;
    regs[0] = f2.x; regs[1] = f2.y;
  #endif
}

CUDA_DEVICE
void ldGlobalVec(const float* ptr, float regs[1]) {
  #if defined(__NVCC__) || defined(__CUDACC__)
    asm volatile ("ld.ca.global.f32 {%0}, [%1];" :
                  "=f"(regs[0]) : "l"(ptr));
  #elif defined(__HIPCC__)
    regs[0] = *ptr;
  #endif
}

CUDA_DEVICE
void ldGlobalVec(const float* ptr, float* regs, uint len) {
  switch(len) {
    case 1:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("ld.ca.global.f32 {%0}, [%1];" :
                    "=f"(regs[0]) : "l"(ptr));
    #elif defined(__HIPCC__)
      regs[0] = *ptr;
    #endif
      break;
    case 2:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("ld.ca.global.v2.f32 {%0, %1}, [%2];" :
                    "=f"(regs[0]), "=f"(regs[1]) : "l"(ptr));
    #elif defined(__HIPCC__)
    {
      float2 f2 = *(float2*)ptr; 
      regs[0] = f2.x; regs[1] = f2.y;
    }
    #endif
      break;
    case 4:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("ld.ca.global.v4.f32 {%0, %1, %2, %3}, [%4];" :
                    "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]) : "l"(ptr));
    #elif defined(__HIPCC__)
    {
      float4 f4 = *(float4*)ptr; 
      regs[0] = f4.x; regs[1] = f4.y; regs[2] = f4.z; regs[3] = f4.w;
    }
    #endif
      break;
  }
}

//int loads
CUDA_DEVICE
void ldGlobalVec(const int* ptr, int4& vec) {
  vec = *(int4*)ptr;
}

//double loads
CUDA_DEVICE
void ldGlobalVec(const double* ptr, double4& vec) {
  vec = *(double4*)ptr;
}

CUDA_DEVICE
void sharedStore(float* ptr, float val) {
#if defined(__NVCC__) || defined(__CUDACC__)
  asm volatile ("st.shared.f32 [%0], {%1};\n" :: "l"(ptr), "f"(val));
#elif defined(__HIPCC__)
  *ptr = val;
#endif
}