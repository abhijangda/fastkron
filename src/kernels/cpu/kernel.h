#include "kernels/params.h"
#include "utils/utils.h"

#include <immintrin.h>

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

  const uint32_t TileM = 2;
  const uint32_t TileK = 1024;
  const uint32_t TileQ = 16;
  const uint32_t TileP = P;

  const uint32_t RegM = TileM;
  const uint32_t RegK = 8; //MIN(TileK, 8);
  const uint32_t RegQ = 8; //MIN(TileQ, 8);

  const uint32_t YRegs = RegM * RegK * RegQ;
  const uint32_t XRegs = RegM * RegK;
  const uint32_t FRegs = RegQ;

  const uint32_t VectorLen = 8; //AVX256 length

  assert (RegQ % VectorLen == 0);

  const uint32_t VecRegK = RegK;
  const uint32_t VecRegM = RegM; //(RegK < VectorLen) ? VectorLen/RegK : RegM;
  const uint32_t VecRegQ = RegQ/VectorLen;

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
      
      //TODO: Different vector lengths. AVX512, AVX256, AVX, SSE4.2, no vector based on underlying architecture

      __m256 yReg[VecRegM][VecRegK][VecRegQ];

      // YRegisters<ElemT, > yReg;
      for (uint32_t ym = 0; ym < VecRegM; ym++) {
      for (uint32_t yk = 0; yk < VecRegK; yk++) {
      for (uint32_t yq = 0; yq < VecRegQ; yq++) {
        yReg[ym][yk][yq] =  _mm256_setzero_ps();
      }}}

      __m256 XReg[VecRegM][VecRegK];
      __m256 FReg[VecRegQ];

      #pragma unroll
      for (uint32_t p = 0; p < P; p++) {
        #pragma unroll
        for (uint32_t em = 0; em < VecRegM; em++) {
          #pragma unroll
          for (uint32_t ek = 0; ek < RegK; ek++) {
            XReg[em][ek] = _mm256_broadcast_ss(XTile.data(m + em, k + ek*P, 0)+p);
          }
        }

        // for (uint32_t em = 0; em < VecRegM; em++) {
        //   __m256i addr;
        //   #pragma unroll
        //   for (uint32_t ek = 0; ek < RegK; ek++) { //TODO: += VecRegK
        //     addr = _mm256_insert_epi32(addr, ( + p) - XTile.ptr, ek);
        //   }
        //   XReg[em][0] = _mm256_i32gather_ps(XTile.ptr, addr, sizeof(float));
        // }

        #pragma unroll
        for (uint32_t rq = 0; rq < VecRegQ; rq++) {
          FReg[rq] = _mm256_loadu_ps(F.data<ElemT>(p, tileQ + q + rq * VectorLen, OpF));
        }

        #pragma unroll
        for (uint32_t rm = 0; rm < VecRegM; rm++) {
        #pragma unroll
        for (uint32_t rk = 0; rk < VecRegK; rk++) {
        #pragma unroll
        for (uint32_t rq = 0; rq < VecRegQ; rq++) {
          yReg[rm][rk][rq] = _mm256_fmadd_ps(XReg[rm][rk], FReg[rq], yReg[rm][rk][rq]);
        }}}
      }
    
      const uint32_t XTileSlices = TileK/P;
      const uint32_t XSlices     = K/P;

      for (uint32_t rm = 0; rm < VecRegM; rm++) {
      for (uint32_t rk = 0; rk < RegK; rk++) {
      for (uint32_t rq = 0; rq < VecRegQ; rq++) {
        __m256 reg = yReg[rm][rk][rq];
        float buf[8];
        _mm256_storeu_ps(buf, reg);
        for (uint32_t elem = 0; elem < VectorLen; elem++) {
          const uint32_t cacheK = (rq*VectorLen + elem + q) * XTileSlices + rk + k/P;
          uint32_t memK = (cacheK/XTileSlices) * XSlices +
                          (tileK/TileK) * XTileSlices +
                          cacheK % XTileSlices;

          if (TileQ != Q) {
            const uint32_t QTiles = Q/TileQ;
            memK += (tileQ/TileQ) * (Y.n()/QTiles);
          }

          Y.set<ElemT>(tileM + m + rm, memK, fastKronOp_N, buf[elem]);
        }
      }}}
    }}}
  }}}
}