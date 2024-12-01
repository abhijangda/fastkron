#include "gtest/gtest.h"
#include "testBase.h"

template<typename T>
bool test(FastKronMMType mmtype, const uint M, const uint N, const uint K, const uint NUM_KP_MATS, 
          uint* KP_MAT_N, uint* KP_MAT_K, fastKronOp opx, fastKronOp opfs,
          uint32_t batchCountZ, uint32_t batchCountX, uint32_t batchCountF, uint32_t batchCountY,
          T alpha, T beta, bool tune, bool isforward) {
  return run<T>(mmtype, M, N, K, NUM_KP_MATS, KP_MAT_N, KP_MAT_K, opx, opfs, batchCountZ, batchCountX, batchCountF, batchCountY, alpha, beta, 1, 0, false, 1, 1, 1, 1, true, true, tune, getTestBackend(), isforward, false);
}

#define GENERAL_TEST_NN(MMType, M, MinFacs, MaxFacs, P, Q, Type, Tune, IsForward, BatchZ, BatchX, BatchF, BatchY) \
  TEST(EXPAND(TEST_BACKEND,Fusion), MMType##_##Type##_##M##x##MinFacs##_##MaxFacs##_##P##x##Q##_##Tune##_##IsForward##_##BatchZ##x##BatchX##x##BatchF##x##BatchY##_##NN) { \
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
    result = result and test(MMType, M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_N, fastKronOp_N, BatchZ, BatchX, BatchF, BatchY, alpha, beta, Tune, IsForward);\
    if (!result) abort();\
  }\
  EXPECT_TRUE(result);\
}

#define CONTIGUOUS_TEST_NN(MMType, M1, MinN, MaxN, P, Q, Tune, IsForward) \
  GENERAL_TEST_NN(MMType, M1, MinN, MaxN, P, Q, float, Tune, IsForward, 1, 1, 1, 1); \
  GENERAL_TEST_NN(MMType, M1, MinN, MaxN, P, Q, double, Tune, IsForward, 1, 1, 1, 1); \

#define CONTIGUOUS_TEST_MMTYPE_NN(MinN, MaxN, P, Q, Tune, IsForward) \
  CONTIGUOUS_TEST_NN(MKM, 16, MinN, MaxN, P, Q, Tune, IsForward); \
  CONTIGUOUS_TEST_NN(KMM, 1, MinN, MaxN, P, Q, Tune, IsForward); \
  CONTIGUOUS_TEST_NN(KMM, 3, MinN, MaxN, P, Q, Tune, IsForward); \
  CONTIGUOUS_TEST_NN(KMM, 16, MinN, MaxN, P, Q, Tune, IsForward); \
  CONTIGUOUS_TEST_NN(KMM, 20, MinN, MaxN, P, Q, Tune, IsForward); \



#define STRIDED_BATCHED_TEST_NN(MinN, MaxN, P, Q, Tune, IsForward, BatchZ, BatchX, BatchF, BatchY) \
  GENERAL_TEST_NN(KMM, 1, MinN, MaxN, P, Q, float, Tune, IsForward, BatchZ, BatchX, BatchF, BatchY); \
  GENERAL_TEST_NN(MKM, 16, MinN, MaxN, P, Q, float, Tune, IsForward, BatchZ, BatchX, BatchF, BatchY); \
  GENERAL_TEST_NN(MKM, 16, MinN, MaxN, P, Q, double, Tune, IsForward, BatchZ, BatchX, BatchF, BatchY); \
  
CONTIGUOUS_TEST_MMTYPE_NN(1, 9, 1, 1, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(5, 9, 2, 2, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 7, 3, 3, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 7, 4, 4, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 6, 5, 5, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 6, 6, 6, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 5, 8, 8, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 4, 12, 12, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 4, 16, 16, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 4, 24, 24, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 3, 31, 31, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 3, 32, 32, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 3, 50, 50, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 3, 55, 55, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 3, 62, 62, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 3, 64, 64, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 3, 127, 127, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 2, 128, 128, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 2, 129, 129, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 2, 255, 255, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 2, 297, 297, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 2, 384, 384, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 2, 505, 505, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 2, 512, 512, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 2, 739, 739, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 2, 1024, 1024, false, false);

CONTIGUOUS_TEST_MMTYPE_NN(1, 5, 1, 4, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 5, 5, 1, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 7, 2, 4, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 3, 31, 63, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 3, 63, 31, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 2, 297, 127, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 2, 127, 297, false, false);
CONTIGUOUS_TEST_MMTYPE_NN(1, 2, 936, 505, false, false);

CONTIGUOUS_TEST_MMTYPE_NN(1, 3, 128, 128, true, false);
CONTIGUOUS_TEST_MMTYPE_NN(2, 3, 32, 32, true, false);
CONTIGUOUS_TEST_MMTYPE_NN(3, 5, 16, 8, true, true);

CONTIGUOUS_TEST_MMTYPE_NN(1, 4, 8, 32, true, true);

STRIDED_BATCHED_TEST_NN(1, 3, 128, 128, true, false, 2, 2, 2, 2);
STRIDED_BATCHED_TEST_NN(1, 3, 32, 16,  false, false, 2, 2, 2, 2);
STRIDED_BATCHED_TEST_NN(3, 4, 32, 32, true, false, 2, 1, 2, 2);
STRIDED_BATCHED_TEST_NN(1, 5, 12, 16, true, false, 2, 2, 1, 1);
STRIDED_BATCHED_TEST_NN(3, 4, 32,  64,   false, false, 2, 1, 2, 2);

STRIDED_BATCHED_TEST_NN(3, 4, 5,  5,   true, true, 2, 1, 2, 2);

#define GENERAL_DISTINCT_FACTORS_TEST_NN(MMType, M, FacCase, Type, Tune, IsForward, BatchZ, BatchX, BatchF, BatchY)\
  TEST(EXPAND(TEST_BACKEND, Fusion), MMType##_##Type##_##M##x##FacCase##_##Tune##_##IsForward##_##BatchZ##x##BatchX##x##BatchY##_##NN) {\
    uint Facs, *P, *Q;\
    if (FacCase == 0) {\
      Facs = 3;\
      P = new uint[Facs];\
      Q = new uint[Facs];\
      uint P_[] = {2, 5, 6};\
      uint Q_[] = {3, 2, 4};\
      memcpy(P, P_, sizeof(uint)*Facs);\
      memcpy(Q, Q_, sizeof(uint)*Facs);\
    } else if (FacCase == 1) {\
      Facs = 2;\
      P = new uint[Facs];\
      Q = new uint[Facs];\
      uint P_[] = {14, 19};\
      uint Q_[] = {15, 13};\
      memcpy(P, P_, sizeof(uint)*Facs);\
      memcpy(Q, Q_, sizeof(uint)*Facs);\
    } else if (FacCase == 2) {\
      Facs = 4;\
      P = new uint[Facs];\
      Q = new uint[Facs];\
      uint P_[] = {14, 19, 34, 21};\
      uint Q_[] = {15, 13, 23, 34};\
      memcpy(P, P_, sizeof(uint)*Facs);\
      memcpy(Q, Q_, sizeof(uint)*Facs);\
    }\
    uint N = 1, K = 1;\
    for (uint i = 0; i < (uint)Facs; i++) {\
      N *= Q[i];\
      K *= P[i];\
    }\
    Type alpha = IsForward ? 1.0f : 2.0f;\
    Type beta = IsForward ? 0.0f : 1.0f;\
    bool result = test(MMType, M, N, K, Facs, Q, P, fastKronOp_N, fastKronOp_N, BatchZ, BatchX, BatchF, BatchY, alpha, beta, Tune, IsForward);\
    delete[] P;\
    delete[] Q;\
    EXPECT_TRUE(result);\
  }\

#define DISTINCT_FACTORS_TEST_NN(MMType, FacCase, BatchZ, BatchX, BatchF, BatchY) \
  GENERAL_DISTINCT_FACTORS_TEST_NN(MMType, 16, FacCase, float, false, false, BatchZ, BatchX, BatchF, BatchY);\
  GENERAL_DISTINCT_FACTORS_TEST_NN(MMType, 3, FacCase, double, false, false, BatchZ, BatchX, BatchF, BatchY);\
  GENERAL_DISTINCT_FACTORS_TEST_NN(MMType, 16, FacCase, float, true, false, BatchZ, BatchX, BatchF, BatchY);\
  GENERAL_DISTINCT_FACTORS_TEST_NN(MMType, 3, FacCase, double, false, true, BatchZ, BatchX, BatchF, BatchY);\
  GENERAL_DISTINCT_FACTORS_TEST_NN(MMType, 3, FacCase, float, true, true, BatchZ, BatchX, BatchF, BatchY);\



DISTINCT_FACTORS_TEST_NN(MKM, 0, 1, 1, 1, 1);
DISTINCT_FACTORS_TEST_NN(MKM, 1, 1, 1, 1, 1);
DISTINCT_FACTORS_TEST_NN(MKM, 2, 1, 1, 1, 1);

DISTINCT_FACTORS_TEST_NN(KMM, 0, 1, 1, 1, 1);
DISTINCT_FACTORS_TEST_NN(KMM, 1, 1, 1, 1, 1);
DISTINCT_FACTORS_TEST_NN(KMM, 2, 1, 1, 1, 1);

DISTINCT_FACTORS_TEST_NN(MKM, 0, 2, 1, 2, 2);
DISTINCT_FACTORS_TEST_NN(MKM, 1, 2, 2, 2, 2);
DISTINCT_FACTORS_TEST_NN(MKM, 2, 2, 2, 2, 2);

DISTINCT_FACTORS_TEST_NN(KMM, 0, 2, 2, 2, 2);
DISTINCT_FACTORS_TEST_NN(KMM, 1, 2, 2, 2, 2);
DISTINCT_FACTORS_TEST_NN(KMM, 2, 2, 2, 2, 2);