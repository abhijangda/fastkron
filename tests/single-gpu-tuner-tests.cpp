#include "gtest/gtest.h"
#include "testBase.h"

#define SINGLE_GPU_NO_FUSION_TUNER_TEST(M, Facs, FacSize, Type) \
  TEST(SingleGPUNoFusionTunerTest, Type##_##M##x##Facs##x##FacSize##_) { \
  uint KP_MAT_N[Facs];\
  uint KP_MAT_K[Facs];\
  uint N = 1;\
  uint K = 1;\
  for (uint i = 0; i < (uint)Facs; i++) {\
    N *= FacSize;\
    K *= FacSize;\
    KP_MAT_K[i] = KP_MAT_N[i] = FacSize;\
  }\
  bool b = run<Type>(M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_N, fastKronOp_N, \
  1, 0, false, 1, 1, 1, 1, true, true, true, fastKronBackend_CUDA, false);\
  EXPECT_TRUE(b);\
}

SINGLE_GPU_NO_FUSION_TUNER_TEST(512, 4, 16, float)

SINGLE_GPU_NO_FUSION_TUNER_TEST(512, 3, 64, float)

// SINGLE_GPU_NO_FUSION_TUNER_TEST(1, 3, 32, float)

// SINGLE_GPU_NO_FUSION_TUNER_TEST(4, 2, 64, float)