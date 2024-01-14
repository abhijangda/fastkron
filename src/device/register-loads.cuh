#include "config.h"

#pragma once

/***float loads***/
CUDA_DEVICE 
void ldGlobalVec(const float4* ptr, float regs[4]) {
  asm volatile ("ld.ca.global.v4.f32 {%0, %1, %2, %3}, [%4];" :
                "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]) : "l"(ptr));
}

CUDA_DEVICE
void ldGlobalVec(const float2* ptr, float regs[2]) {
  asm volatile ("ld.ca.global.v2.f32 {%0, %1}, [%2];" :
                 "=f"(regs[0]), "=f"(regs[1]) : "l"(ptr));
}

CUDA_DEVICE
void ldGlobalVec(const float* ptr, float regs[1]) {
  asm volatile ("ld.ca.global.f32 {%0}, [%1];" :
                "=f"(regs[0]) : "l"(ptr));
}

CUDA_DEVICE
void ldGlobalVec(const float* ptr, float* regs, uint len) {
  switch(len) {
    case 1:
      asm volatile ("ld.ca.global.f32 {%0}, [%1];" :
                    "=f"(regs[0]) : "l"(ptr));
      break;
    case 2:
      asm volatile ("ld.ca.global.v2.f32 {%0, %1}, [%2];" :
                    "=f"(regs[0]), "=f"(regs[1]) : "l"(ptr));
      break;
    case 4:
      asm volatile ("ld.ca.global.v4.f32 {%0, %1, %2, %3}, [%4];" :
                    "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]) : "l"(ptr));
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
  asm volatile ("st.shared.f32 [%0], {%1};\n" :: "l"(ptr), "f"(val));
}