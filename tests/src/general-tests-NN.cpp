#include "gtest/gtest.h"
#include "testBase.h"

#define GENERAL_TEST_NN(M, MinFacs, MaxFacs, P, Q, Type, Tune) \
  TEST(EXPAND(TEST_BACKEND,Fusion), Type##_##M##x##MinFacs##_##MaxFacs##_##P##x##Q##_##Tune##_##NN) { \
  bool result = true;\
  for (uint Facs = MinFacs; Facs <= MaxFacs; Facs++) {\
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
    result = result and run<Type>(M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_N, fastKronOp_N, (Type)2.0f, (Type)3.0f, 1, 0, false, 1, 1, 1, 1, true, true, Tune, getTestBackend(), false);\
    if (!result) abort();\
  }\
  EXPECT_TRUE(result);\
}

#define FLOAT_DOUBLE_TEST_NN(MinN, MaxN, P, Q, Tune) \
  GENERAL_TEST_NN(16, MinN, MaxN, P, Q, float, Tune); \
  GENERAL_TEST_NN(16, MinN, MaxN, P, Q, double, Tune);


FLOAT_DOUBLE_TEST_NN(1, 10, 2, 2, false);
FLOAT_DOUBLE_TEST_NN(1, 8, 3, 3, false);
FLOAT_DOUBLE_TEST_NN(1, 8, 4, 4, false);
FLOAT_DOUBLE_TEST_NN(1, 7, 5, 5, false);
FLOAT_DOUBLE_TEST_NN(1, 7, 6, 6, false);
FLOAT_DOUBLE_TEST_NN(1, 6, 8, 8, false);
FLOAT_DOUBLE_TEST_NN(1, 5, 12, 12, false);
FLOAT_DOUBLE_TEST_NN(1, 5, 16, 16, false);
FLOAT_DOUBLE_TEST_NN(1, 5, 24, 24, false);
FLOAT_DOUBLE_TEST_NN(1, 4, 31, 31, false);
FLOAT_DOUBLE_TEST_NN(1, 4, 32, 32, false);
FLOAT_DOUBLE_TEST_NN(1, 3, 50, 50, false);
FLOAT_DOUBLE_TEST_NN(1, 3, 55, 55, false);
FLOAT_DOUBLE_TEST_NN(1, 3, 62, 62, false);
FLOAT_DOUBLE_TEST_NN(1, 3, 64, 64, false);
FLOAT_DOUBLE_TEST_NN(1, 3, 127, 127, false);
FLOAT_DOUBLE_TEST_NN(1, 3, 128, 128, false);
FLOAT_DOUBLE_TEST_NN(1, 3, 129, 129, false);
FLOAT_DOUBLE_TEST_NN(1, 3, 255, 255, false);
FLOAT_DOUBLE_TEST_NN(1, 2, 297, 297, false);
FLOAT_DOUBLE_TEST_NN(1, 2, 384, 384, false);
FLOAT_DOUBLE_TEST_NN(1, 2, 505, 505, false);
FLOAT_DOUBLE_TEST_NN(1, 2, 512, 512, false);
FLOAT_DOUBLE_TEST_NN(1, 2, 739, 739, false);
FLOAT_DOUBLE_TEST_NN(1, 2, 1024, 1024, false);


FLOAT_DOUBLE_TEST_NN(1, 10, 2, 4, false);
FLOAT_DOUBLE_TEST_NN(1, 4, 31, 63, false);
FLOAT_DOUBLE_TEST_NN(1, 4, 63, 31, false);
FLOAT_DOUBLE_TEST_NN(1, 3, 297, 127, false);
FLOAT_DOUBLE_TEST_NN(1, 3, 127, 297, false);
FLOAT_DOUBLE_TEST_NN(1, 2, 936, 505, false);

// FLOAT_DOUBLE_TEST_NN(1, 3, 128, 128, int);
FLOAT_DOUBLE_TEST_NN(1, 3, 128, 128, true);
FLOAT_DOUBLE_TEST_NN(3, 4, 32, 32, true);
FLOAT_DOUBLE_TEST_NN(6, 8, 4, 4, true);