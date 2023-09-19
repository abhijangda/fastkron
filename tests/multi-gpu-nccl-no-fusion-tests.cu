#include "gtest/gtest.h"
#include "testBase.h"

#define MULTI_GPU_NCCL_NO_FUSION_TEST(M, Facs, FacSize, GM, GK, KronBatch, Type) \
  TEST(MultiGpuNCCL, Type##_##M##x##Facs##x##FacSize##x##GM##x##GK##x##KronBatch##_) { \
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

MULTI_GPU_NCCL_NO_FUSION_TEST(4, 5, 8, 2, 2, 4, float);

MULTI_GPU_NCCL_NO_FUSION_TEST(2, 3, 16, 1, 2, 1, float);
MULTI_GPU_NCCL_NO_FUSION_TEST(2, 3, 16, 1, 4, 2, float);
MULTI_GPU_NCCL_NO_FUSION_TEST(2, 4, 16, 1, 4, 3, float);
MULTI_GPU_NCCL_NO_FUSION_TEST(2, 4, 16, 2, 2, 3, float);

MULTI_GPU_NCCL_NO_FUSION_TEST(4, 4, 32, 2, 2, 3, float);

MULTI_GPU_NCCL_NO_FUSION_TEST(4, 4, 64, 4, 1, 4, float);