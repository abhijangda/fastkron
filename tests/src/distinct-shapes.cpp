#include "gtest/gtest.h"
#include "testBase.h"

TEST(EXPAND(TEST_BACKEND,DistinctShapes), Case1) {
  uint KP_MAT_N[] = {16,32,8,32};
  uint KP_MAT_K[] = {8,8,16,8};
  uint N = 1;
  uint K = 1;
  for (uint i = 0; i < (uint)4; i++) {
    N *= KP_MAT_N[i];
    K *= KP_MAT_K[i];
  }
  bool b = run<float>(128, N, K, 4, KP_MAT_N, KP_MAT_K, fastKronOp_N, fastKronOp_N, 1, 0, false, 1, 1, 1, 1, true, false, true, getTestBackend(), false);
  EXPECT_TRUE(b);
}