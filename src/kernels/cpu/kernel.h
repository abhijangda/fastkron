#include "kernels/params.h"
#include "utils/utils.h"

#include <immintrin.h>

#pragma once

inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {
  __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
  __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
  __t0 = _mm256_unpacklo_ps(row0, row1);
  __t1 = _mm256_unpackhi_ps(row0, row1);
  __t2 = _mm256_unpacklo_ps(row2, row3);
  __t3 = _mm256_unpackhi_ps(row2, row3);
  __t4 = _mm256_unpacklo_ps(row4, row5);
  __t5 = _mm256_unpackhi_ps(row4, row5);
  __t6 = _mm256_unpacklo_ps(row6, row7);
  __t7 = _mm256_unpackhi_ps(row6, row7);
  __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
  __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
  __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
  __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
  __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
  __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
  __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
  __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
  row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
  row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
  row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
  row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
  row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
  row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
  row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
  row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

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
  const uint32_t TileK = 2048;
  const uint32_t TileQ = 32;
  const uint32_t TileP = 32;

  const uint32_t RegM = 1;
  const uint32_t RegK = 8; //MIN(TileK, 8);
  const uint32_t RegQ = 16; //MIN(TileQ, 8);

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
                            P, P, //TODO: setting this to P because XTile.data is not right for GPU backend
                            X);

    ElemT tileBuff[TileM][TileK/P][TileQ];

    for (uint32_t tileP = 0; tileP < P; tileP += TileP) {
      for (uint32_t m = 0; m < TileM; m += RegM) {
      for (uint32_t q = 0; q < TileQ; q += RegQ) {
      for (uint32_t k = 0; k < TileK; k += RegK * P) {
        //TODO: Different vector lengths. AVX512, AVX256, AVX, SSE4.2, no vector based on underlying architecture
        __m256 yReg[VecRegM][VecRegK][VecRegQ];

        if (tileP == 0) {
          // YRegisters<ElemT, > yReg;
          for (uint32_t ym = 0; ym < VecRegM; ym++) {
          for (uint32_t yk = 0; yk < VecRegK; yk++) {
          for (uint32_t yq = 0; yq < VecRegQ; yq++) {
            yReg[ym][yk][yq] =  _mm256_setzero_ps();
          }}}
        } else {
          for (uint32_t ym = 0; ym < VecRegM; ym++) {
          for (uint32_t yk = 0; yk < VecRegK; yk++) {
          for (uint32_t yq = 0; yq < VecRegQ; yq++) {
            yReg[ym][yk][yq] = _mm256_loadu_ps(&tileBuff[m+ym][k/P+yk][q+yq*VectorLen]);
          }}}
        }

        #pragma unroll 2
        for (uint32_t p = 0; p < TileP; p++) {
          __m256 XReg[VecRegM][VecRegK];
          __m256 FReg[VecRegQ];
          #pragma unroll
          for (uint32_t em = 0; em < VecRegM; em++) {
            #pragma unroll
            for (uint32_t ek = 0; ek < RegK; ek++) {
              XReg[em][ek] = _mm256_broadcast_ss(XTile.data(m + em, k + ek*P, 0) + tileP + p);
          }}

          #pragma unroll
          for (uint32_t rq = 0; rq < VecRegQ; rq++) {
            FReg[rq] = _mm256_loadu_ps(F.data<ElemT>(tileP + p, tileQ + q + rq * VectorLen, OpF));
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

        if (tileP < P - TileP) {
          for (uint32_t ym = 0; ym < VecRegM; ym++) {
          for (uint32_t yk = 0; yk < VecRegK; yk++) {
          for (uint32_t yq = 0; yq < VecRegQ; yq++) {
            _mm256_storeu_ps(&tileBuff[m+ym][k/P+yk][q+yq*VectorLen], yReg[ym][yk][yq]);
          }}}
        } else {
          const uint32_t XTileSlices = TileK/P;
          const uint32_t XSlices     = K/P;

          for (uint32_t rm = 0; rm < VecRegM; rm++) {
          for (uint32_t rq = 0; rq < VecRegQ; rq++) {
            transpose8_ps(yReg[rm][0][rq], yReg[rm][1][rq], yReg[rm][2][rq], 
                          yReg[rm][3][rq], yReg[rm][4][rq], yReg[rm][5][rq],
                          yReg[rm][6][rq], yReg[rm][7][rq]);

          for (uint32_t rk = 0; rk < RegK; rk++) {
            __m256 reg = yReg[rm][rk][rq];
            const uint32_t cacheK = (rq*VectorLen + q + rk) * XTileSlices + rk/VectorLen + k/P;
            uint32_t memK = (cacheK/XTileSlices) * XSlices +
                            (tileK/TileK) * XTileSlices +
                            cacheK % XTileSlices;

            if (TileQ != Q) {
              const uint32_t QTiles = Q/TileQ;
              memK += (tileQ/TileQ) * (Y.n()/QTiles);
            }

            _mm256_storeu_ps(Y.data<ElemT>(tileM + m + rm, memK, fastKronOp_N), reg);
          }}}
        }
      }}}
    }
  }}}
}