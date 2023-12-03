#include "gtest/gtest.h"
#include "testBase.h"

#define MULTI_GPU_TUNER_TEST(M, Facs, FacSize, GM, GK, KronBatch, Type) \
  TEST(MultiGpuTuner, Type##_##M##x##Facs##x##FacSize##x##GM##x##GK##x##KronBatch##_) { \
  uint KP_MAT_N[Facs];\
  uint KP_MAT_K[Facs];\
  uint N = 1;\
  uint K = 1;\
  for (uint i = 0; i < (uint)Facs; i++) {\
    N *= FacSize;\
    K *= FacSize;\
    KP_MAT_K[i] = KP_MAT_N[i] = FacSize;\
  }\
  bool b = run<Type>(M, N, K, Facs, KP_MAT_N, KP_MAT_K, 0, 0, false, GM, GK, GM*GK, KronBatch, true, true, true, false);\
  EXPECT_TRUE(b);\
}

MULTI_GPU_TUNER_TEST(512, 5, 16, 2, 1, 4, float);
MULTI_GPU_TUNER_TEST(512, 4, 16, 4, 2, 3, float);
MULTI_GPU_TUNER_TEST(512, 4, 16, 2, 2, 2, float);