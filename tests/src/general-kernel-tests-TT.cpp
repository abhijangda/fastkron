#include "gtest/gtest.h"
#include "testBase.h"

#define FUSION_TEST(M, MinFacs, MaxFacs, P, Q, Type) \
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
    result = result and run<Type>(M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_T, fastKronOp_T, 1, 0, false, 1, 1, 1, 1, true, true, true, getTestBackend(), false);\
    if (!result) abort();\
  }\
  EXPECT_TRUE(result);\
}

#define M_ 16
FUSION_TEST(M_, 1, 1, 2, 2, float);
FUSION_TEST(M_, 1, 1, 3, 3, float);
FUSION_TEST(M_, 1, 1, 4, 4, float);
FUSION_TEST(M_, 1, 1, 5, 5, float);
FUSION_TEST(M_, 1, 1, 6, 6, float);
FUSION_TEST(M_, 1, 1, 8, 8, float);
FUSION_TEST(M_, 1, 1, 12, 12, float);
FUSION_TEST(M_, 1, 1, 16, 16, float);
FUSION_TEST(M_, 1, 1, 24, 24, float);
FUSION_TEST(M_, 1, 1, 31, 31, float);
FUSION_TEST(M_, 1, 1, 32, 32, float);
FUSION_TEST(M_, 1, 1, 50, 50, float);
FUSION_TEST(M_, 1, 1, 55, 55, float);
FUSION_TEST(M_, 1, 1, 62, 62, float);
FUSION_TEST(M_, 1, 1, 64, 64, float);
FUSION_TEST(M_, 1, 1, 127, 127, float);
FUSION_TEST(M_, 1, 1, 128, 128, float);
FUSION_TEST(M_, 1, 1, 129, 129, float);
FUSION_TEST(M_, 1, 1, 255, 255, float);
FUSION_TEST(M_, 1, 1, 297, 297, float);
FUSION_TEST(M_, 1, 1, 384, 384, float);
FUSION_TEST(M_, 1, 1, 505, 505, float);
FUSION_TEST(M_, 1, 1, 512, 512, float);
FUSION_TEST(M_, 1, 1, 739, 739, float);
FUSION_TEST(M_, 1, 1, 1024, 1024, float);


FUSION_TEST(M_, 1, 1, 31, 63, float);
FUSION_TEST(M_, 1, 1, 63, 31, float);
FUSION_TEST(M_, 1, 1, 297, 127, float);
FUSION_TEST(M_, 1, 1, 127, 297, float);
FUSION_TEST(M_, 1, 1, 936, 505, float);