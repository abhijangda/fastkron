#include "gtest/gtest.h"
#include "testBase.h"

#define NON_SQUARE(M, Facs, P, Q, Type) \
  TEST(EXPAND(TEST_BACKEND,OddShape), Type##_##M##x##Facs##x##P##x##Q##_) { \
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
  bool b = run<Type>(M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_N, fastKronOp_N, 1, 0, false, 1, 1, 1, 1, true, false, true, getTestBackend(), false);\
  EXPECT_TRUE(b);\
}

NON_SQUARE(12, 2, 31, 16, float)
NON_SQUARE(8, 2, 16, 31, float)
NON_SQUARE(6, 4, 31, 31, float)