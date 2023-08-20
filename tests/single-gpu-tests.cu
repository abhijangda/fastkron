#include "gtest/gtest.h"
#include "testBase.h"

#define SINGLE_GPU_TEST(M, Facs, FacSize, Type, VecType) \
  TEST(SingleGPU, Type##_##M##x##Facs##x##FacSize##_) { \
  uint KP_MAT_N[Facs];\
  uint KP_MAT_K[Facs];\
  uint N = 1;\
  uint K = 1;\
  for (uint i = 0; i < (uint)Facs; i++) {\
    N *= FacSize;\
    K *= FacSize;\
    KP_MAT_K[i] = KP_MAT_N[i] = FacSize;\
  }\
  bool b = run<Type, VecType>(M, N, K, Facs, KP_MAT_N, KP_MAT_K, 1, 0, false, 0, 0, 0, 1, true, false);\
  EXPECT_TRUE(b);\
}

//FacSize 2
SINGLE_GPU_TEST(1, 7, 2, float, float4);
SINGLE_GPU_TEST(1, 8, 2, float, float4);
SINGLE_GPU_TEST(1, 10, 2, float, float4);
SINGLE_GPU_TEST(1, 15, 2, float, float4);
SINGLE_GPU_TEST(1, 20, 2, float, float4);

//FacSize 4
SINGLE_GPU_TEST(1, 4, 4, float, float4);
SINGLE_GPU_TEST(1, 6, 4, float, float4);
SINGLE_GPU_TEST(1, 8, 4, float, float4);
SINGLE_GPU_TEST(1, 9, 4, float, float4);
SINGLE_GPU_TEST(1, 10, 4, float, float4);

//FacSize 8
SINGLE_GPU_TEST(1, 4, 8, float, float4);
SINGLE_GPU_TEST(1, 5, 8, float, float4);
SINGLE_GPU_TEST(1, 6, 8, float, float4);
SINGLE_GPU_TEST(1, 7, 8, float, float4);
SINGLE_GPU_TEST(1, 8, 8, float, float4);

//FacSize 16
SINGLE_GPU_TEST(1, 2, 16, float, float4);
SINGLE_GPU_TEST(1, 3, 16, float, float4);
SINGLE_GPU_TEST(1, 4, 16, float, float4);
SINGLE_GPU_TEST(1, 5, 16, float, float4);
SINGLE_GPU_TEST(1, 6, 16, float, float4);

//FacSize 32
SINGLE_GPU_TEST(1, 2, 32, float, float4);
SINGLE_GPU_TEST(1, 3, 32, float, float4);
SINGLE_GPU_TEST(1, 4, 32, float, float4);
SINGLE_GPU_TEST(1, 5, 32, float, float4);

//FacSize 64
SINGLE_GPU_TEST(1, 2, 64, float, float4);
SINGLE_GPU_TEST(1, 3, 64, float, float4);
SINGLE_GPU_TEST(1, 4, 64, float, float4);

//FacSize 128
SINGLE_GPU_TEST(1, 2, 128, float, float4);
SINGLE_GPU_TEST(1, 3, 128, float, float4);

// //FacSize 256
// SINGLE_GPU_TEST(1, 2, 128, float, float4);
// SINGLE_GPU_TEST(1, 3, 128, float, float4);