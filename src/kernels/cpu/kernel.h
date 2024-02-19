#include "kernels/params.h"

#pragma once

template<typename ElemT, typename Vec2T, typename Vec4T,
         uint MaxQ, uint MaxP, uint FusedFacs, fastKronOp OpX, fastKronOp OpF>
void cpuKernel(KernelParams<FusedFacs> params,
               FusedParams<FusedFacs> fusedParams,
               DistributedParams distParams,
               EpilogueParams epilogueParams) {
  Matrix X = params.problem.x();
  Matrix Y = params.problem.y();
  Factor F = params.problem.f(0);

  uint32_t K = X.n();
  uint32_t P = F.p();

  for (int m = 0; m < X.m(); m++) {
    for (int q = 0; q < F.q(); q++) {
      for (int k = 0; k < K; k += P) {
        uint slice = k/P;
        ElemT acc = (ElemT)0;
        for (int p = 0; p < P; p++) {
          acc += X.at<ElemT>(m, k + p, OpX) * F.at<ElemT>(p, q, OpF);
        }

        Y.set(m, slice + (K/P) * q, fastKronOp_N, acc);
      }
    }
  }
}