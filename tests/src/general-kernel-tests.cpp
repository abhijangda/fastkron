#include "gtest/gtest.h"
#include "testBase.h"

#define GENERAL_TEST(M, MinFacs, MaxFacs, P, Q, Type) \
  TEST(EXPAND(TEST_BACKEND,Fusion), Type##_##M##x##MinFacs##_##MaxFacs##_##P##x##Q##_##NN) { \
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
    result = result and run<Type>(M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_N, fastKronOp_N, 1, 0, false, 1, 1, 1, 1, true, true, true, getTestBackend(), false);\
    if (!result) abort();\
  }\
  EXPECT_TRUE(result);\
}

#define FLOAT_DOUBLE_TEST(MinN, MaxN, P, Q) \
  GENERAL_TEST(16, MinN, MaxN, P, Q, float); \
  // GENERAL_TEST(16, MinN, MaxN, P, Q, double);


FLOAT_DOUBLE_TEST(1, 10, 2, 2);
FLOAT_DOUBLE_TEST(1, 8, 3, 3);
FLOAT_DOUBLE_TEST(1, 8, 4, 4);
FLOAT_DOUBLE_TEST(1, 7, 5, 5);
FLOAT_DOUBLE_TEST(1, 7, 6, 6);
FLOAT_DOUBLE_TEST(1, 6, 8, 8);
FLOAT_DOUBLE_TEST(1, 5, 12, 12);
FLOAT_DOUBLE_TEST(1, 5, 16, 16);
FLOAT_DOUBLE_TEST(1, 5, 24, 24);
FLOAT_DOUBLE_TEST(1, 4, 31, 31);
FLOAT_DOUBLE_TEST(1, 4, 32, 32);
FLOAT_DOUBLE_TEST(1, 3, 50, 50);
FLOAT_DOUBLE_TEST(1, 3, 55, 55);
FLOAT_DOUBLE_TEST(1, 3, 62, 62);
FLOAT_DOUBLE_TEST(1, 3, 64, 64);
FLOAT_DOUBLE_TEST(1, 3, 127, 127);
FLOAT_DOUBLE_TEST(1, 3, 128, 128);
FLOAT_DOUBLE_TEST(1, 3, 129, 129);
FLOAT_DOUBLE_TEST(1, 3, 255, 255);
FLOAT_DOUBLE_TEST(1, 2, 297, 297);
FLOAT_DOUBLE_TEST(1, 2, 384, 384);
FLOAT_DOUBLE_TEST(1, 2, 505, 505);
FLOAT_DOUBLE_TEST(1, 2, 512, 512);
FLOAT_DOUBLE_TEST(1, 2, 739, 739);
FLOAT_DOUBLE_TEST(1, 2, 1024, 1024);


FLOAT_DOUBLE_TEST(1, 4, 31, 63);
FLOAT_DOUBLE_TEST(1, 4, 63, 31);
FLOAT_DOUBLE_TEST(1, 3, 297, 127);
FLOAT_DOUBLE_TEST(1, 3, 127, 297);
FLOAT_DOUBLE_TEST(1, 2, 936, 505);

// FLOAT_DOUBLE_TEST(1, 3, 128, 128, int);