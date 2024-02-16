#include "gtest/gtest.h"
#include "testBase.h"

#define MULTI_GPU_NON_SQUARE_TEST(M, Facs, P, Q, GM, GK, KronBatch, Type) \
  TEST(MultiGpuNonSquare, Type##_##M##x##Facs##x##P##x##Q##_##GM##x##GK##x##KronBatch##_) { \
  uint KP_MAT_N[Facs];\
  uint KP_MAT_K[Facs];\
  uint N = 1;\
  uint K = 1;\
  for (uint i = 0; i < (uint)Facs; i++) {\
    N *= Q;\
    K *= P;\
    KP_MAT_K[i] = P;\
    KP_MAT_N[i] = Q;\
  }\
  bool b = run<Type>(M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_N, fastKronOp_N, 1, 0, false, 1, 1, 1, 1, true, false, true, fastKronBackend_CUDA, false);\
  EXPECT_TRUE(b);\
}

MULTI_GPU_NON_SQUARE_TEST(18, 5, 8, 32, 2, 1, 5, float);
MULTI_GPU_NON_SQUARE_TEST(14, 5, 8, 32, 1, 2, 3, float);
MULTI_GPU_NON_SQUARE_TEST(18, 5, 8, 32, 4, 4, 3, float);

MULTI_GPU_NON_SQUARE_TEST(18, 4, 64, 16, 2, 1, 4, float);
MULTI_GPU_NON_SQUARE_TEST(14, 4, 64, 16, 1, 2, 3, float);
MULTI_GPU_NON_SQUARE_TEST(18, 4, 64, 16, 4, 4, 3, float);