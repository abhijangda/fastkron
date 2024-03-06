#include "kernels/params.h"
#include "utils/utils.h"

#include <immintrin.h>
#include <cstring>
#include <cstdlib>
#include <omp.h>

#pragma once

// https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
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
         uint MaxQ, uint MaxP, uint TileP, uint TileQ, uint TileK,
         uint TileM, uint FusedFacs, uint RegK, uint RegQ,
         fastKronOp OpX, fastKronOp OpF>
void cpuKernel(KernelParams<FusedFacs> params,
               FusedParams<FusedFacs> fusedParams,
               DistributedParams distParams,
               EpilogueParams epilogueParams) {
  Matrix X = params.problem.x();
  Matrix Y = params.problem.y();
  Factor F = params.problem.f(0);

  const uint32_t K = X.n();
  const uint32_t P = F.p();
  const uint32_t Q = F.q();

  // const uint32_t TileM = 1;
  // const uint32_t TileK = 4096;
  // const uint32_t TileQ = 128;
  // const uint32_t TileP = 128;

  const uint32_t RegM = TileM;
  // const uint32_t RegK = 16; //MIN(TileK, 8);
  // const uint32_t RegQ = 8; //MIN(TileQ, 8);

  const uint32_t YRegs = RegM * RegK * RegQ;
  const uint32_t XRegs = RegM * RegK;
  const uint32_t FRegs = RegQ;

  const uint32_t VectorLen = 8; //AVX256 length

  static_assert(RegK % VectorLen == 0);
  static_assert(TileK % RegK == 0);
  static_assert(TileQ % RegQ == 0);
  assert(FusedFacs == 1 || (FusedFacs > 1 && P <= TileP && Q <= TileQ && P == Q));
  //For Transpose load loop to TileX
  assert ((TileK/P) % VectorLen == 0);

  const uint32_t VecRegK = RegK/VectorLen;
  const uint32_t VecRegM = RegM; //(RegK < VectorLen) ? VectorLen/RegK : RegM;
  const uint32_t VecRegQ = RegQ;

  uint threads = omp_get_max_threads();
  
  const size_t SzTileX = TileM*TileP*(TileK/P);
  //TODO: Allocate this in fastKron_initBackend
  static ElemT* TileXs[96] = {nullptr};
  static ElemT* TileYs[96] = {nullptr};

  if (TileXs[0] == nullptr) {
    for (int i = 0; i < 96; i++)  {
      TileXs[i] = (ElemT*)aligned_alloc(8192, SzTileX * sizeof(ElemT));
      TileYs[i] = (ElemT*)aligned_alloc(8192, TileM * TileQ * TileK/P * sizeof(ElemT));
    }
  }

  #pragma omp parallel for collapse(3)
  for (uint32_t tileM = 0; tileM < X.m(); tileM += TileM) {
  for (uint32_t tileQ = 0; tileQ < Q    ; tileQ += TileQ) {
  for (uint32_t tileK = 0; tileK < K    ; tileK += TileK) {
    Slice<ElemT, OpX> XTile(tileM, tileK, 
                            (TileM == 1) ? 1 : MIN(TileM, X.m() - tileM), TileK,
                            P, P, //TODO: setting this to P because XTile.data is not right for GPU backend
                            X);

    const uint tid = omp_get_thread_num();
    ElemT* tileBuff = TileYs[tid];

    for (int fac = FusedFacs - 1; fac >= 0; fac--) {
      ElemT* TileX = TileXs[tid];
      //Transpose X data and store to TileX to reduce TLB misses
      for (uint32_t tileP = 0; tileP < P; tileP += TileP) {
        for (uint32_t m = 0; m < XTile.m(); m++) {
          uint32_t NumSlices = VectorLen;
          for (uint32_t k = 0; k < TileK; k += NumSlices * P) {
            for (uint32_t p = 0; p < ROUNDDOWN(TileP, VectorLen); p += VectorLen) {
              __m256 slices[VectorLen];
              for (uint32_t sliceIdx = 0; sliceIdx < NumSlices; sliceIdx++) {
                const ElemT* ptr = (fac == FusedFacs - 1) ? XTile.data(m, k + sliceIdx*P + tileP + p, 0) :
                                                            &tileBuff[m * TileK + k + sliceIdx*P + tileP + p];
                slices[sliceIdx] = _mm256_loadu_ps(ptr);
              }

              transpose8_ps(slices[0], slices[1], slices[2], 
                            slices[3], slices[4], slices[5],
                            slices[6], slices[7]);

              for (uint32_t pp = 0; pp < VectorLen; pp++) {
                _mm256_storeu_ps(&TileX[m*TileP*(TileK/P) + (p + pp)*(TileK/P) + k/P], slices[pp]);
              }
            }
          
            for (uint32_t p = ROUNDDOWN(TileP, VectorLen); p < TileP; p++) {
              for (uint32_t sliceIdx = 0; sliceIdx < NumSlices; sliceIdx++) {
                const ElemT* ptr = (fac == FusedFacs - 1) ? XTile.data(m, k + sliceIdx*P + tileP + p, 0) :
                                                            &tileBuff[m * TileK + k + sliceIdx*P + tileP + p];
                TileX[m*TileP*(TileK/P) + p*(TileK/P) + k/P + sliceIdx] = *ptr;
              }
            }
          }
        }

        ElemT TileF[TileP][TileQ];
        Factor F = params.problem.f(fac);

        for (int p = 0; p < TileP; p++) {
          memcpy(&TileF[p][0], F.data<ElemT>(tileP + p, tileQ, OpF), TileQ * sizeof(ElemT));
        }
        
        for (uint32_t m = 0; m < XTile.m(); m += RegM) {
        for (uint32_t q = 0; q < TileQ; q += RegQ) {
        for (uint32_t k = 0; k < TileK/P * TileP; k += RegK * TileP) {
          //TODO: Different vector lengths. AVX512, AVX256, AVX, SSE4.2, no vector based on underlying architecture
          __m256 yReg[VecRegM][VecRegQ][VecRegK];

          if (tileP == 0) {
            // YRegisters<ElemT, > yReg;
            for (uint32_t ym = 0; ym < VecRegM; ym++) {
            for (uint32_t yq = 0; yq < VecRegQ; yq++) {
            for (uint32_t yk = 0; yk < VecRegK; yk++) {
              yReg[ym][yq][yk] =  _mm256_setzero_ps();
            }}}
          } else {
            for (uint32_t ym = 0; ym < VecRegM; ym++) {
            for (uint32_t yk = 0; yk < VecRegK; yk++) {
            for (uint32_t yq = 0; yq < VecRegQ; yq++) {
              yReg[ym][yq][yk] = _mm256_loadu_ps(&tileBuff[(m+ym)*TileQ*(TileK/P) + (q+yq)*(TileK/P) + k/TileP+yk*VectorLen]);
            }}}
          }

          for (uint32_t p = 0; p < TileP; p++) {
            __m256 XReg[VecRegM][VecRegK];
            __m256 FReg[VecRegQ];

            if (VecRegM == 1 && VecRegK == 2 && VecRegQ == 4) {
              ElemT* xptr = &TileX[(m)*TileP*(TileK/P) + p * (TileK/P) + k/TileP + 0];
              __m256 x0 = _mm256_loadu_ps(xptr);
              __m256 x1 = _mm256_loadu_ps(xptr + 1*VectorLen);

              ElemT* fptr = &TileF[p][q];
              __m256 f0 = _mm256_broadcast_ss(fptr);
              __m256 f1 = _mm256_broadcast_ss(fptr + 1);
              __m256 f2 = _mm256_broadcast_ss(fptr + 2);
              __m256 f3 = _mm256_broadcast_ss(fptr + 3);


              yReg[0][0][0] = _mm256_fmadd_ps(x0, f0, yReg[0][0][0]);
              yReg[0][1][0] = _mm256_fmadd_ps(x0, f1, yReg[0][1][0]);
              yReg[0][2][0] = _mm256_fmadd_ps(x0, f2, yReg[0][2][0]);
              yReg[0][3][0] = _mm256_fmadd_ps(x0, f3, yReg[0][3][0]);
              yReg[0][0][1] = _mm256_fmadd_ps(x1, f0, yReg[0][0][1]);
              yReg[0][1][1] = _mm256_fmadd_ps(x1, f1, yReg[0][1][1]);
              yReg[0][2][1] = _mm256_fmadd_ps(x1, f2, yReg[0][2][1]);
              yReg[0][3][1] = _mm256_fmadd_ps(x1, f3, yReg[0][3][1]);

              // yReg[0][0][0] = y00;
              // yReg[0][0][1] = y01;
              // yReg[0][0][2] = y02;
              // yReg[0][0][3] = y03;
              // yReg[0][1][0] = y10;
              // yReg[0][1][1] = y11;
              // yReg[0][1][2] = y12;
              // yReg[0][1][3] = y13;
            } else {
              #pragma unroll
              for (uint32_t em = 0; em < VecRegM; em++) {
                #pragma unroll
                for (uint32_t ek = 0; ek < VecRegK; ek++) {
                  XReg[em][ek] = _mm256_loadu_ps(&TileX[(m + em)*TileP*(TileK/P) + p * (TileK/P) + k/TileP + ek*VectorLen]);
              }}

              #pragma unroll
              for (uint32_t rq = 0; rq < VecRegQ; rq++) {
                FReg[rq] = _mm256_broadcast_ss(&TileF[p][q+rq]);
              }

              #pragma unroll
              for (uint32_t rm = 0; rm < VecRegM; rm++) {
              #pragma unroll
              for (uint32_t rk = 0; rk < VecRegK; rk++) {
              #pragma unroll
              for (uint32_t rq = 0; rq < VecRegQ; rq++) {
                yReg[rm][rq][rk] = _mm256_fmadd_ps(XReg[rm][rk], FReg[rq], yReg[rm][rq][rk]);
              }}}
            }
          }

          if (tileP < P - TileP) {
            for (uint32_t ym = 0; ym < VecRegM; ym++) {
            for (uint32_t yq = 0; yq < VecRegQ; yq++) {
            for (uint32_t yk = 0; yk < VecRegK; yk++) {
              _mm256_storeu_ps(&tileBuff[(m+ym)*TileQ*(TileK/P) + (q+yq)*(TileK/P) + k/TileP+yk*VectorLen], yReg[ym][yq][yk]);
            }}}
          } else {
            const uint32_t XTileSlices = TileK/P;
            const uint32_t XSlices     = K/P;

            for (uint32_t rm = 0; rm < VecRegM; rm++) {
            for (uint32_t rq = 0; rq < VecRegQ; rq++) {
            for (uint32_t rk = 0; rk < VecRegK; rk++) {
              __m256 reg = yReg[rm][rq][rk];
              const uint32_t cacheK = (rq + q) * XTileSlices + rk*VectorLen + k/TileP;
              if (fac > 0) {
                if (m + rm < XTile.m()) {
                  // ElemT b[8]; _mm256_storeu_ps(b, reg); printf("%f %f %f %f\n", b[0], b[1], b[2], b[3]);
                  _mm256_storeu_ps(&tileBuff[(m+rm)*TileK + cacheK], reg);
                }
              } else {
                //TODO: Need to fix
                uint32_t memK;
                if (FusedFacs > 1) {
                  uint32_t xshCol = cacheK;
                  //Scale shared mem slice idx to global mem idx
                  uint32_t glSlice = (xshCol/XTileSlices)*XSlices;
                  //Scale shared fused slice to global mem
                  uint32_t sliceElem = ((xshCol%XTileSlices)/fusedParams.XShFusedSlices)*fusedParams.XglFusedSlices;
                  //Elem idx in Fused Slice
                  uint32_t elem = (tileK/TileK) * fusedParams.XShFusedSlices + xshCol%fusedParams.XShFusedSlices;
                  memK = glSlice + sliceElem + elem; 
                } else {
                  memK = (cacheK/XTileSlices) * XSlices +
                          (tileK/TileK) * XTileSlices +
                          cacheK % XTileSlices;

                  if (TileQ != Q) {
                    const uint32_t QTiles = Q/TileQ;
                    memK += (tileQ/TileQ) * (Y.n()/QTiles);
                  }
                }
                if (m + rm < XTile.m())
                  _mm256_storeu_ps(Y.data<ElemT>(tileM + m + rm, memK, fastKronOp_N), reg);
              }
            }}}
          }
        }}}
      }
    }
  }}}

  // free(TileXs);
}