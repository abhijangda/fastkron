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

  const uint32_t K = X.n();
  const uint32_t P = 64;//F.p();
  const uint32_t Q = 64;

  const uint32_t TileM = 4;
  const uint32_t TileK = 1024;
  const uint32_t TileQ = 16;
  const uint32_t TileP = P;

  const uint32_t RegM = MIN(TileM, 2);
  const uint32_t RegK = MIN(TileK, 8);
  const uint32_t RegQ = MIN(TileQ, 4);
  const uint32_t RegP = MIN(TileP, 4);

  #pragma omp parallel for collapse(3)
  for (uint32_t tileM = 0; tileM < X.m(); tileM += TileM) {
  for (uint32_t tileQ = 0; tileQ < Q    ; tileQ += TileQ) {
  for (uint32_t tileK = 0; tileK < K    ; tileK += TileK) {
    Slice<ElemT, OpX> XTile(tileM, tileK, 
                          (TileM == 1) ? 1 : MIN(TileM, X.m() - tileM), TileK,
                          P, TileP,
                          X);
    for (uint32_t m = 0; m < TileM; m += RegM) {
    for (uint32_t q = 0; q < TileQ; q += RegQ) {
    for (uint32_t k = 0; k < TileK; k += RegK * P) {

      ElemT yReg[TileM][RegK][RegQ] = {0};
      // YRegisters<ElemT, > yReg;

      for (uint32_t rm = 0; rm < RegM; rm++) {
      for (uint32_t rq = 0; rq < RegQ; rq++) {
      for (uint32_t rk = 0; rk < RegK; rk++) {
        for (uint32_t p = 0; p < P; p++) {
          yReg[rm][rk][rq] += XTile.data(m + rm, k + rk*P, 0)[p] *
                              F.at<ElemT>(p, tileQ + q + rq, OpF);
        }
      }}}
    
      const uint32_t XTileSlices = TileK/P;
      const uint32_t XSlices     = K/P;

      for (uint32_t rm = 0; rm < RegM; rm++) {
      for (uint32_t rq = 0; rq < RegQ; rq++) {
      for (uint32_t rk = 0; rk < RegK; rk++) {
        const uint32_t cacheK = (rq + q) * XTileSlices + rk + k/P;
        uint32_t memK = (cacheK/XTileSlices) * XSlices +
                        (tileK/TileK) * XTileSlices +
                        cacheK % XTileSlices;

        if (TileQ != Q) {
          const uint32_t QTiles = Q/TileQ;
          memK += (tileQ/TileQ) * (Y.n()/QTiles);
        }

        Y.set(tileM + m + rm, memK, fastKronOp_N, yReg[rm][rk][rq]);
      }}}
    }}}
  }}}
}