#include "gtest/gtest.h"
#include "testBase.h"

#define MULTI_GPU_NO_FUSION_TEST(M, Facs, FacSize, GM, GK, KronBatch, Type) \
  TEST(MultiGpuNoFusion, Type##_##M##x##Facs##x##FacSize##x##GM##x##GK##x##KronBatch##_) { \
  uint KP_MAT_N[Facs];\
  uint KP_MAT_K[Facs];\
  uint N = 1;\
  uint K = 1;\
  for (uint i = 0; i < (uint)Facs; i++) {\
    N *= FacSize;\
    K *= FacSize;\
    KP_MAT_K[i] = KP_MAT_N[i] = FacSize;\
  }\
  bool b = run<Type>(M, N, K, Facs, KP_MAT_N, KP_MAT_K, 0, 0, false, GM, GK, GM*GK, KronBatch, true, false, false, false);\
  EXPECT_TRUE(b);\
}

MULTI_GPU_NO_FUSION_TEST(4, 4, 64, 2, 1, 4, float);
MULTI_GPU_NO_FUSION_TEST(8, 4, 64, 4, 2, 3, float);
MULTI_GPU_NO_FUSION_TEST(8, 4, 64, 2, 2, 2, float);

MULTI_GPU_NO_FUSION_TEST(2, 3, 128, 1, 4, 2, float);
MULTI_GPU_NO_FUSION_TEST(4, 4, 128, 4, 4, 3, float);