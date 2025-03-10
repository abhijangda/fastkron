#include "gtest/gtest.h"
#include "testBase.h"

#define GENERAL_TEST_TT(MMType, M, MinFacs, MaxFacs, P, Q, Type, Tune, IsForward, BatchZ, BatchX, BatchF, BatchY) \
  TEST(EXPAND(TEST_BACKEND,Fusion), MMType##_##Type##_##M##x##MinFacs##_##MaxFacs##_##P##x##Q##_##Tune##_##IsForward##_##BatchZ##x##BatchX##x##BatchF##x##BatchY##_##TT) { \
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
    Type alpha = IsForward ? 1.0f : 2.0f;\
    Type beta = IsForward ? 0.0f : 1.0f;\
    result = result and run<Type>(MMType, M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_T, fastKronOp_T, BatchZ,BatchX,BatchF,BatchY,alpha, beta, 1, 0, false, 1, 1, 1, 1, true, true, Tune, getTestBackend(), false, false);\
    if (!result) abort();\
  }\
  EXPECT_TRUE(result);\
}

#define CONTIGUOUS_TEST_TT(MinN, MaxN, P, Q, Type, Tune, IsForward) \
    GENERAL_TEST_TT(MKM, 16, MinN, MaxN, P, Q, Type, Tune, IsForward, 1, 1, 1, 1); \
    GENERAL_TEST_TT(MKM, 13, MinN, MaxN, P, Q, Type, Tune, IsForward, 1, 1, 1, 1); \
    GENERAL_TEST_TT(KMM, 1, MinN, MaxN, P, Q, Type, Tune, IsForward, 1, 1, 1, 1); \
    GENERAL_TEST_TT(KMM, 17, MinN, MaxN, P, Q, Type, Tune, IsForward, 1, 1, 1, 1); \

#define STRIDED_BATCHED_TEST_TT(MinN, MaxN, P, Q, Type, Tune, IsForward, BatchZ, BatchX, BatchF, BatchY) \
    GENERAL_TEST_TT(MKM, 16, MinN, MaxN, P, Q, Type, Tune, IsForward, BatchZ, BatchX, BatchF, BatchY); \
    GENERAL_TEST_TT(KMM, 3, MinN, MaxN, P, Q, Type, Tune, IsForward, BatchZ, BatchX, BatchF, BatchY);

CONTIGUOUS_TEST_TT(1, 10, 2, 1, float, false, false);
CONTIGUOUS_TEST_TT(1, 9, 1, 6, float, false, false);
CONTIGUOUS_TEST_TT(1, 10, 2, 2, float, false, false);
CONTIGUOUS_TEST_TT(1, 6, 3, 3, float, false, false);
CONTIGUOUS_TEST_TT(1, 6, 4, 4, float, false, false);
CONTIGUOUS_TEST_TT(1, 5, 5, 5, float, false, false);
CONTIGUOUS_TEST_TT(1, 5, 6, 6, float, false, false);
CONTIGUOUS_TEST_TT(1, 5, 8, 8, float, false, false);
CONTIGUOUS_TEST_TT(1, 5, 12, 12, float, false, false);
CONTIGUOUS_TEST_TT(1, 5, 16, 16, float, false, false);
CONTIGUOUS_TEST_TT(1, 5, 24, 24, float, false, false);
CONTIGUOUS_TEST_TT(1, 4, 31, 31, float, false, false);
CONTIGUOUS_TEST_TT(1, 4, 32, 32, float, false, false);
CONTIGUOUS_TEST_TT(1, 3, 50, 50, float, false, false);
CONTIGUOUS_TEST_TT(1, 3, 55, 55, float, false, false);
CONTIGUOUS_TEST_TT(1, 3, 62, 62, float, false, false);
CONTIGUOUS_TEST_TT(1, 3, 64, 64, float, false, false);
CONTIGUOUS_TEST_TT(1, 3, 127, 127, float, false, false);
CONTIGUOUS_TEST_TT(1, 3, 128, 128, float, false, false);
CONTIGUOUS_TEST_TT(1, 3, 129, 129, float, false, false);
CONTIGUOUS_TEST_TT(1, 2, 255, 255, float, false, false);
CONTIGUOUS_TEST_TT(1, 2, 297, 297, float, false, false);
CONTIGUOUS_TEST_TT(1, 2, 384, 384, float, false, false);
CONTIGUOUS_TEST_TT(1, 1, 505, 505, float, false, false);
CONTIGUOUS_TEST_TT(1, 1, 512, 512, float, false, false);
CONTIGUOUS_TEST_TT(1, 1, 739, 739, float, false, false);
CONTIGUOUS_TEST_TT(1, 1, 1024, 1024, float, false, false);

CONTIGUOUS_TEST_TT(1, 5, 8, 2, float, false, false);
CONTIGUOUS_TEST_TT(1, 3, 31, 63, float, false, false);
CONTIGUOUS_TEST_TT(1, 3, 63, 31, float, false, false);
CONTIGUOUS_TEST_TT(1, 2, 297, 127, float, false, false);
CONTIGUOUS_TEST_TT(1, 2, 127, 297, float, false, false);
CONTIGUOUS_TEST_TT(1, 2, 936, 505, float, false, false);

CONTIGUOUS_TEST_TT(1, 3, 128, 128, float, true, false);
CONTIGUOUS_TEST_TT(3, 4, 32, 32, float, true, false);
CONTIGUOUS_TEST_TT(3, 5, 18, 8, float, true, false);

CONTIGUOUS_TEST_TT(1, 4, 8, 32, float, true, true);

STRIDED_BATCHED_TEST_TT(1, 3, 128, 128, float, true, false, 2, 2, 2, 2);
STRIDED_BATCHED_TEST_TT(3, 4, 32,  32,  float, true, false, 2, 1, 2, 2);
STRIDED_BATCHED_TEST_TT(1, 3, 64,  64,  float, false, false, 2, 2, 2, 2);
STRIDED_BATCHED_TEST_TT(1, 5, 12,  12,  float, true, false, 2, 2, 1, 1);
STRIDED_BATCHED_TEST_TT(3, 4, 5,   5,   float, false, false, 2, 1, 2, 2);

STRIDED_BATCHED_TEST_TT(1, 3, 128, 128, float, true, true, 2, 2, 2, 2);
