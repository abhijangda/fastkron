#include "gtest/gtest.h"
#include "testBase.h"

#define NON_SQUARE(M, Facs, P, Q, Type) \
  TEST(EXPAND(TEST_BACKEND,NonSquare), Type##_##M##x##Facs##x##P##x##Q##_) { \
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

NON_SQUARE(11, 4, 8, 16, float)
NON_SQUARE(12, 5, 8, 16, float)

NON_SQUARE(12, 3, 128, 32, float)

NON_SQUARE(11, 3, 32, 16, float)
NON_SQUARE(12, 4, 32, 16, float)