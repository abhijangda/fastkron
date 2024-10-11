#include "gtest/gtest.h"
#include "testBase.h"

#define GENERAL_TEST_TT(M, MinFacs, MaxFacs, P, Q, Type, Tune, BatchZ, BatchX, BatchF, BatchY) \
  TEST(EXPAND(TEST_BACKEND,Fusion), Type##_##M##x##MinFacs##_##MaxFacs##_##P##x##Q##_##Tune##_##BatchZ##x##BatchX##x##BatchF##x##BatchY##TT) { \
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
    result = result and run<Type>(M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_T, fastKronOp_T, BatchZ,BatchX,BatchF,BatchY,(Type)2.0f, (Type)3.0f, 1, 0, false, 1, 1, 1, 1, true, true, Tune, getTestBackend(), false);\
    if (!result) abort();\
  }\
  EXPECT_TRUE(result);\
}

#define M_ 16

#define CONTIGUOUS_TEST_TT(M,MinN, MaxN, P, Q, Type, Tune) \
    GENERAL_TEST_TT(M, MinN, MaxN, P, Q, Type, Tune, 1, 1, 1, 1);

#define STRIDED_BATCHED_TEST_TT(M, MinN, MaxN, P, Q, Type, Tune, BatchZ, BatchX, BatchF, BatchY) \
    GENERAL_TEST_TT(M, MinN, MaxN, P, Q, Type, Tune, BatchZ, BatchX, BatchF, BatchY);

CONTIGUOUS_TEST_TT(M_, 1, 10, 2, 1, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 10, 1, 6, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 10, 2, 2, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 8, 3, 3, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 8, 4, 4, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 7, 5, 5, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 7, 6, 6, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 6, 8, 8, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 5, 12, 12, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 5, 16, 16, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 5, 24, 24, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 4, 31, 31, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 4, 32, 32, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 3, 50, 50, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 3, 55, 55, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 3, 62, 62, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 3, 64, 64, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 3, 127, 127, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 3, 128, 128, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 3, 129, 129, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 2, 255, 255, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 2, 297, 297, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 2, 384, 384, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 1, 505, 505, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 1, 512, 512, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 1, 739, 739, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 1, 1024, 1024, float, false);

CONTIGUOUS_TEST_TT(M_, 1, 5, 8, 2, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 3, 31, 63, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 3, 63, 31, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 2, 297, 127, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 2, 127, 297, float, false);
CONTIGUOUS_TEST_TT(M_, 1, 2, 936, 505, float, false);

CONTIGUOUS_TEST_TT(M_, 1, 3, 128, 128, float, true);
CONTIGUOUS_TEST_TT(M_, 3, 4, 32, 32, float, true);
CONTIGUOUS_TEST_TT(M_, 6, 8, 4, 4, float, true);

STRIDED_BATCHED_TEST_TT(M_, 1, 3, 128, 128, float, true, 2, 2, 2, 2);
STRIDED_BATCHED_TEST_TT(M_, 3, 4, 32,  32,  float, true, 2, 1, 2, 2);
STRIDED_BATCHED_TEST_TT(M_, 1, 5, 12,  12,  float, true, 2, 2, 1, 1);