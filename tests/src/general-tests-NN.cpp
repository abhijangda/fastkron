#include "gtest/gtest.h"
#include "testBase.h"

#define GENERAL_TEST_NN(MMType, M, MinFacs, MaxFacs, P, Q, Type, Tune, BatchZ, BatchX, BatchF, BatchY) \
  TEST(EXPAND(TEST_BACKEND,Fusion), Type##_##M##x##MinFacs##_##MaxFacs##_##P##x##Q##_##Tune##_##BatchZ##x##BatchX##x##BatchF##x##BatchY##_##NN) { \
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
    result = result and run<Type>(MMType, M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_N, fastKronOp_N, BatchZ,BatchX,BatchF,BatchY,(Type)2.0f, (Type)3.0f, 1, 0, false, 1, 1, 1, 1, true, true, Tune, getTestBackend(), false);\
    if (!result) abort();\
  }\
  EXPECT_TRUE(result);\
}

#define CONTIGUOUS_TEST_NN(MMType, MinN, MaxN, P, Q, Tune) \
  GENERAL_TEST_NN(MMType, 16, MinN, MaxN, P, Q, float, Tune, 1, 1, 1, 1); \
  GENERAL_TEST_NN(MMType, 16, MinN, MaxN, P, Q, double, Tune, 1, 1, 1, 1);

#define STRIDED_BATCHED_TEST_NN(MMType, MinN, MaxN, P, Q, Tune, BatchZ, BatchX, BatchF, BatchY) \
  GENERAL_TEST_NN(MMType, 16, MinN, MaxN, P, Q, float, Tune, BatchZ, BatchX, BatchF, BatchY); \
  GENERAL_TEST_NN(MMType, 16, MinN, MaxN, P, Q, double, Tune, BatchZ, BatchX, BatchF, BatchY);

CONTIGUOUS_TEST_NN(MKM, 1, 10, 1, 1, false);
CONTIGUOUS_TEST_NN(MKM, 1, 10, 2, 2, false);
CONTIGUOUS_TEST_NN(MKM, 1, 8, 3, 3, false);
CONTIGUOUS_TEST_NN(MKM, 1, 8, 4, 4, false);
CONTIGUOUS_TEST_NN(MKM, 1, 7, 5, 5, false);
CONTIGUOUS_TEST_NN(MKM, 1, 7, 6, 6, false);
CONTIGUOUS_TEST_NN(MKM, 1, 6, 8, 8, false);
CONTIGUOUS_TEST_NN(MKM, 1, 5, 12, 12, false);
CONTIGUOUS_TEST_NN(MKM, 1, 5, 16, 16, false);
CONTIGUOUS_TEST_NN(MKM, 1, 5, 24, 24, false);
CONTIGUOUS_TEST_NN(MKM, 1, 4, 31, 31, false);
CONTIGUOUS_TEST_NN(MKM, 1, 4, 32, 32, false);
CONTIGUOUS_TEST_NN(MKM, 1, 3, 50, 50, false);
CONTIGUOUS_TEST_NN(MKM, 1, 3, 55, 55, false);
CONTIGUOUS_TEST_NN(MKM, 1, 3, 62, 62, false);
CONTIGUOUS_TEST_NN(MKM, 1, 3, 64, 64, false);
CONTIGUOUS_TEST_NN(MKM, 1, 3, 127, 127, false);
CONTIGUOUS_TEST_NN(MKM, 1, 3, 128, 128, false);
CONTIGUOUS_TEST_NN(MKM, 1, 3, 129, 129, false);
CONTIGUOUS_TEST_NN(MKM, 1, 3, 255, 255, false);
CONTIGUOUS_TEST_NN(MKM, 1, 2, 297, 297, false);
CONTIGUOUS_TEST_NN(MKM, 1, 2, 384, 384, false);
CONTIGUOUS_TEST_NN(MKM, 1, 2, 505, 505, false);
CONTIGUOUS_TEST_NN(MKM, 1, 2, 512, 512, false);
CONTIGUOUS_TEST_NN(MKM, 1, 2, 739, 739, false);
CONTIGUOUS_TEST_NN(MKM, 1, 2, 1024, 1024, false);

CONTIGUOUS_TEST_NN(MKM, 1, 10, 1, 4, false);
CONTIGUOUS_TEST_NN(MKM, 1, 10, 5, 1, false);
CONTIGUOUS_TEST_NN(MKM, 1, 10, 2, 4, false);
CONTIGUOUS_TEST_NN(MKM, 1, 4, 31, 63, false);
CONTIGUOUS_TEST_NN(MKM, 1, 4, 63, 31, false);
CONTIGUOUS_TEST_NN(MKM, 1, 3, 297, 127, false);
CONTIGUOUS_TEST_NN(MKM, 1, 3, 127, 297, false);
CONTIGUOUS_TEST_NN(MKM, 1, 2, 936, 505, false);

CONTIGUOUS_TEST_NN(MKM, 1, 3, 128, 128, true);
CONTIGUOUS_TEST_NN(MKM, 3, 4, 32, 32, true);
CONTIGUOUS_TEST_NN(MKM, 6, 8, 4, 4, true);

STRIDED_BATCHED_TEST_NN(MKM, 1, 3, 128, 128, true, 2, 2, 2, 2);
STRIDED_BATCHED_TEST_NN(MKM, 1, 3, 64, 64, false, 2, 2, 2, 2);
STRIDED_BATCHED_TEST_NN(MKM, 3, 4, 32,  32,  true, 2, 1, 2, 2);
STRIDED_BATCHED_TEST_NN(MKM, 1, 5, 12,  12,  true, 2, 2, 1, 1);
STRIDED_BATCHED_TEST_NN(MKM, 3, 4, 5,  5,  false, 2, 1, 2, 2);