#include "gtest/gtest.h"
#include "testBase.h"

#define FUSION_TEST(M, MinFacs, MaxFacs, P, Q, Type) \
  TEST(EXPAND(TEST_BACKEND,Fusion), Type##_##M##x##MinFacs##_##MaxFacs##_##P##x##Q##_##TT) { \
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
    result = result and run<Type>(M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_T, fastKronOp_T, 1, 0, false, 1, 1, 1, 1, true, true, true, getTestBackend(), false);\
    if (!result) abort();\
  }\
  EXPECT_TRUE(result);\
}

#define M_ 16
FUSION_TEST(M_, 1, 10, 2, 2, float);
FUSION_TEST(M_, 1, 8, 3, 3, float);
FUSION_TEST(M_, 1, 8, 4, 4, float);
FUSION_TEST(M_, 1, 7, 5, 5, float);
FUSION_TEST(M_, 1, 7, 6, 6, float);
FUSION_TEST(M_, 1, 6, 8, 8, float);
FUSION_TEST(M_, 1, 5, 12, 12, float);
FUSION_TEST(M_, 1, 5, 16, 16, float);
FUSION_TEST(M_, 1, 5, 24, 24, float);
FUSION_TEST(M_, 1, 4, 31, 31, float);
FUSION_TEST(M_, 1, 4, 32, 32, float);
FUSION_TEST(M_, 1, 3, 50, 50, float);
FUSION_TEST(M_, 1, 3, 55, 55, float);
FUSION_TEST(M_, 1, 3, 62, 62, float);
FUSION_TEST(M_, 1, 3, 64, 64, float);
FUSION_TEST(M_, 1, 3, 127, 127, float);
FUSION_TEST(M_, 1, 3, 128, 128, float);
FUSION_TEST(M_, 1, 3, 129, 129, float);
FUSION_TEST(M_, 1, 2, 255, 255, float);
FUSION_TEST(M_, 1, 2, 297, 297, float);
FUSION_TEST(M_, 1, 2, 384, 384, float);
FUSION_TEST(M_, 1, 1, 505, 505, float);
FUSION_TEST(M_, 1, 1, 512, 512, float);
FUSION_TEST(M_, 1, 1, 739, 739, float);
FUSION_TEST(M_, 1, 1, 1024, 1024, float);

FUSION_TEST(M_, 1, 3, 31, 63, float);
FUSION_TEST(M_, 1, 3, 63, 31, float);
FUSION_TEST(M_, 1, 2, 297, 127, float);
FUSION_TEST(M_, 1, 2, 127, 297, float);
FUSION_TEST(M_, 1, 2, 936, 505, float);