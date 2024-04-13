#include "gtest/gtest.h"
#include "testBase.h"

#define MULTI_GPU_DISTINCT_SHAPES_TEST(M, GM, GK, LocalKrons) \
TEST(MultiGPUDistinctShapesTest, GM##_##GK##_##LocalKrons##_) {\
  uint KP_MAT_N[] = {16,8,8,8};\
  uint KP_MAT_K[] = {8,32,16,32};\
  uint N = 1;\
  uint K = 1;\
  for (uint i = 0; i < (uint)4; i++) {\
    N *= KP_MAT_N[i];\
    K *= KP_MAT_K[i];\
  }\
  bool b = run<float>(M, N, K, 4, KP_MAT_N, KP_MAT_K, fastKronOp_N, fastKronOp_N, 1, 0, false, GM, GK, GM*GK, LocalKrons, true, false, true, fastKronBackend_CUDA, false);\
  EXPECT_TRUE(b);\
}

MULTI_GPU_DISTINCT_SHAPES_TEST(256, 1, 2, 1);
MULTI_GPU_DISTINCT_SHAPES_TEST(256, 2, 2, 1);