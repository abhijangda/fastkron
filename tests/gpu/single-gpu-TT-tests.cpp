#include "gtest/gtest.h"
#include "testBase.h"

#define SINGLE_GPU_TT_TEST(M, Facs, FacSize, Type) \
  TEST(SingleGPUTT, Type##_##M##x##Facs##x##FacSize##_) { \
  uint KP_MAT_N[Facs];\
  uint KP_MAT_K[Facs];\
  uint N = 1;\
  uint K = 1;\
  for (uint i = 0; i < (uint)Facs; i++) {\
    N *= FacSize;\
    K *= FacSize;\
    KP_MAT_K[i] = KP_MAT_N[i] = FacSize;\
  }\
  bool b = run<Type>(M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_T, fastKronOp_T, 1, 0, false, 1, 1, 1, 1, true, true, true, fastKronBackend_CUDA, false);\
  EXPECT_TRUE(b);\
}

//FacSize 8
SINGLE_GPU_TT_TEST(11, 4, 8, float);
SINGLE_GPU_TT_TEST(11, 5, 8, float);
SINGLE_GPU_TT_TEST(11, 8, 8, float);

//FacSize 16
SINGLE_GPU_TT_TEST(11, 2, 16, float);
SINGLE_GPU_TT_TEST(11, 5, 16, float);
SINGLE_GPU_TT_TEST(11, 6, 16, float);

//FacSize 32
SINGLE_GPU_TT_TEST(11, 2, 32, float);
SINGLE_GPU_TT_TEST(11, 5, 32, float);

//FacSize 64
SINGLE_GPU_TT_TEST(11, 2, 64, float);
SINGLE_GPU_TT_TEST(11, 4, 64, float);

//FacSize 128
SINGLE_GPU_TT_TEST(11, 2, 128, float);
SINGLE_GPU_TT_TEST(11, 3, 128, float);