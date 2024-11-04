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
  bool b = run<Type>(FastKronMMType::MKM, M, N, K, Facs, KP_MAT_N, KP_MAT_K, fastKronOp_N, fastKronOp_N,1,1,1,1, 1.0f, 0.0f, 0, 0, false, GM, GK, GM*GK, KronBatch, true, true, true, fastKronBackend_CUDA, false);\
  EXPECT_TRUE(b);\
}

MULTI_GPU_TUNER_TEST(128, 5, 16, 2, 1, 5, float);
MULTI_GPU_TUNER_TEST(128, 4, 16, 2, 2, 3, float);