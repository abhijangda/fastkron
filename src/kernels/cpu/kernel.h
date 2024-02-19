#include "kernels/params.h"
#include "utils/utils.h"

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

  const uint32_t K = 64*64;//X.n();
  const uint32_t P = 64;//F.p();
  const uint32_t Q = 64;

  const uint32_t TileM = 4;
  const uint32_t TileK = 1024;
  const uint32_t TileQ = 16;
  const uint32_t TileP = P;

  #pragma omp parallel for collapse(3)
  for (int tileM = 0; tileM < X.m(); tileM += TileM) {
  for (int tileQ = 0; tileQ < Q; tileQ += TileQ) {
  for (int tileK = 0; tileK < K    ; tileK += TileK) {
    Slice<ElemT, OpX> XTile(tileM, tileK, 
                          (TileM == 1) ? 1 : MIN(TileM, X.m() - tileM), TileK,
                          P, TileP,
                          X);
    for (int m = 0; m < TileM; m++) {
    for (int q = 0; q < TileQ; q++) {
    for (int k = 0; k < TileK; k += P) {
      ElemT acc = (ElemT)0;
      for (int p = 0; p < P; p++) {
        acc += XTile.data(m, k, 0)[p] *
               F.at<ElemT>(p, tileQ + q, OpF);
      }

      const uint32_t XTileSlices = TileK/P;
      const uint32_t XSlices     = K/P;

      const uint32_t cacheK = q * XTileSlices + k/P;
      uint32_t memK = (cacheK/XTileSlices) * XSlices +
                      (tileK/TileK) * XTileSlices +
                      cacheK % XTileSlices;

      if (TileQ != Q) {
        const uint32_t QTiles = Q/TileQ;
        memK += (tileQ/TileQ) * (Y.n()/QTiles);
      }

      Y.set(tileM + m, memK, fastKronOp_N, acc);
    }}}
  }}}
}