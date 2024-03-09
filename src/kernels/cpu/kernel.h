#include "kernels/params.h"
#include "utils/utils.h"

#include <immintrin.h>
#include <cstring>
#include <cstdlib>
#include <omp.h>

#pragma once

template<uint32_t VectorLen> class FloatVectorType;

template<uint32_t VectorLen>
inline void transpose(FloatVectorType<VectorLen> rows[VectorLen]) {}
template<> 
inline void transpose(FloatVectorType<8> rows[8]);

template<uint32_t VectorLen>
class FloatVectorType 
{
public:
  void load(const float* ptr);
  void store(float* ptr);
  void zero();
  void broadcast(const float* ptr);
  void fmadd(FloatVectorType<VectorLen>& a, FloatVectorType<VectorLen>& b);
  friend void transpose<VectorLen>(FloatVectorType<VectorLen>[VectorLen]);
};

template<>
class FloatVectorType<8>
{
private:
  __m256 data;
public:
  void load(const float* ptr) {
    data = _mm256_loadu_ps(ptr);
  }

  void store(float* ptr) {
    _mm256_storeu_ps(ptr, data);  
  }
  
  void zero() {
    data = _mm256_setzero_ps();
  }

  void broadcast(const float* ptr) {
    data = _mm256_broadcast_ss(ptr);
  }

  void fmadd(FloatVectorType<8>& a, FloatVectorType<8>& b) {
    data = _mm256_fmadd_ps(a.data, b.data, data);
  }

  friend void transpose<8>(FloatVectorType<8>[8]);
};

template<>
class FloatVectorType<1>
{
private:
  float data;
public:
  void load(const float* ptr) {
    data = *ptr;
  }
  
  void store(float* ptr) {
    *ptr = data;
  }

  void zero() {
    data = 0;
  }

  void broadcast(const float* ptr) {
    load(ptr);
  }

  void fmadd(FloatVectorType<1>& a, FloatVectorType<1>& b) {
    data = a.data*b.data + data;
  }
  
  friend void transpose<1>(FloatVectorType<1>[8]);
};

// https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
template<>
inline void transpose(FloatVectorType<8> rows[8]) {
  __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
  __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
  __t0 = _mm256_unpacklo_ps(rows[0].data, rows[1].data);
  __t1 = _mm256_unpackhi_ps(rows[0].data, rows[1].data);
  __t2 = _mm256_unpacklo_ps(rows[2].data, rows[3].data);
  __t3 = _mm256_unpackhi_ps(rows[2].data, rows[3].data);
  __t4 = _mm256_unpacklo_ps(rows[4].data, rows[5].data);
  __t5 = _mm256_unpackhi_ps(rows[4].data, rows[5].data);
  __t6 = _mm256_unpacklo_ps(rows[6].data, rows[7].data);
  __t7 = _mm256_unpackhi_ps(rows[6].data, rows[7].data);
  __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
  __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
  __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
  __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
  __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
  __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
  __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
  __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
  rows[0].data = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
  rows[1].data = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
  rows[2].data = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
  rows[3].data = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
  rows[4].data = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
  rows[5].data = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
  rows[6].data = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
  rows[7].data = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

template<typename ElemT, uint VectorLen, uint MaxQ, uint MaxP, 
         uint TileP, uint TileQ, uint TileK,
         uint TileM, uint FusedFacs, uint RegK, uint RegQ,
         typename SliceX>
__attribute__((always_inline)) static inline
void vectorMMAAndStore(uint32_t tileM, uint32_t tileK, uint32_t tileP, uint32_t tileQ, uint32_t m, uint32_t q, uint32_t k, uint32_t fac, ElemT* TileX, ElemT* TileF, uint32_t P, uint32_t Q, uint32_t K, SliceX& XTile, ElemT* tileBuff, Matrix& Y, FusedParams<FusedFacs>& fusedParams) {
  //TODO: Different vector lengths. AVX512, AVX256, AVX, SSE4.2, no vector based on underlying architecture
  const uint32_t RegM = TileM;
  const uint32_t VecRegK = RegK/VectorLen;
  const uint32_t VecRegM = RegM; //(RegK < VectorLen) ? VectorLen/RegK : RegM;
  const uint32_t VecRegQ = RegQ;

  using VectorType = FloatVectorType<VectorLen>;
  VectorType yReg[VecRegM][VecRegQ][VecRegK];

  if (tileP == 0) {
    for (uint32_t ym = 0; ym < VecRegM; ym++) {
    for (uint32_t yq = 0; yq < VecRegQ; yq++) {
    for (uint32_t yk = 0; yk < VecRegK; yk++) {
      yReg[ym][yq][yk].zero();
    }}}
  } else {
    for (uint32_t ym = 0; ym < VecRegM; ym++) {
    for (uint32_t yk = 0; yk < VecRegK; yk++) {
    for (uint32_t yq = 0; yq < VecRegQ; yq++) {
      yReg[ym][yq][yk].load(&tileBuff[(m+ym)*TileQ*(TileK/P) + (q+yq)*(TileK/P) + k/TileP+yk*VectorLen]);
    }}}
  }

  if (VectorLen == 8 && VecRegM == 1 && VecRegK == 2 && VecRegQ == 4) {
    for (uint32_t p = 0; p < TileP; p += 2) {
      {
        ElemT* xptr = &TileX[(m)*TileP*(TileK/P) + p * (TileK/P) + k/TileP + 0];
        VectorType x0, x1;

        x0.load(xptr);
        x1.load(xptr + 1*VectorLen);

        ElemT* fptr = &TileF[p*TileQ + q];
        VectorType f0, f1, f2, f3;
        f0.broadcast(fptr);
        f1.broadcast(fptr + 1);
        f2.broadcast(fptr + 2);
        f3.broadcast(fptr + 3);

        _mm_prefetch(&TileX[(m)*TileP*(TileK/P) + (p + 1) * (TileK/P) + k/TileP + 0], _MM_HINT_T1);

        yReg[0][0][0].fmadd(x0, f0);
        yReg[0][1][0].fmadd(x0, f1);
        yReg[0][2][0].fmadd(x0, f2);
        yReg[0][3][0].fmadd(x0, f3);
        yReg[0][0][1].fmadd(x1, f0);
        yReg[0][1][1].fmadd(x1, f1);
        yReg[0][2][1].fmadd(x1, f2);
        yReg[0][3][1].fmadd(x1, f3);
      }
      {
        ElemT* xptr = &TileX[(m)*TileP*(TileK/P) + (p + 1) * (TileK/P) + k/TileP + 0];
        VectorType x0, x1;
        x0.load(xptr);
        x1.load(xptr + 1*VectorLen);

        ElemT* fptr = &TileF[(p + 1)*TileQ + q];
        VectorType f0, f1, f2, f3;
        f0.broadcast(fptr);
        f1.broadcast(fptr + 1);
        f2.broadcast(fptr + 2);
        f3.broadcast(fptr + 3);

        if (p + 2 < TileP) {
          _mm_prefetch(&TileX[(m)*TileP*(TileK/P) + (p + 2) * (TileK/P) + k/TileP + 0], _MM_HINT_T1);
          _mm_prefetch(fptr + 4, _MM_HINT_T1);
        }

        yReg[0][0][0].fmadd(x0, f0);
        yReg[0][1][0].fmadd(x0, f1);
        yReg[0][2][0].fmadd(x0, f2);
        yReg[0][3][0].fmadd(x0, f3);
        yReg[0][0][1].fmadd(x1, f0);
        yReg[0][1][1].fmadd(x1, f1);
        yReg[0][2][1].fmadd(x1, f2);
        yReg[0][3][1].fmadd(x1, f3);

      }
    }
  } else {
    for (uint32_t p = 0; p < TileP; p++) {
      VectorType XReg[VecRegM][VecRegK];
      VectorType FReg[VecRegQ];
      #pragma unroll
      for (uint32_t em = 0; em < VecRegM; em++) {
        #pragma unroll
        for (uint32_t ek = 0; ek < VecRegK; ek++) {
          XReg[em][ek].load(&TileX[(m + em)*TileP*(TileK/P) + p * (TileK/P) + k/TileP + ek*VectorLen]);
      }}

      #pragma unroll
      for (uint32_t rq = 0; rq < VecRegQ; rq++) {
        FReg[rq].broadcast(&TileF[p*TileQ + q + rq]);
      }

      #pragma unroll
      for (uint32_t rm = 0; rm < VecRegM; rm++) {
      #pragma unroll
      for (uint32_t rk = 0; rk < VecRegK; rk++) {
      #pragma unroll
      for (uint32_t rq = 0; rq < VecRegQ; rq++) {
        yReg[rm][rq][rk].fmadd(XReg[rm][rk], FReg[rq]);
      }}}
    }
  }

  if (tileP < P - TileP) {
    for (uint32_t ym = 0; ym < VecRegM; ym++) {
    for (uint32_t yq = 0; yq < VecRegQ; yq++) {
    for (uint32_t yk = 0; yk < VecRegK; yk++) {
      yReg[ym][yq][yk].store(&tileBuff[(m+ym)*TileQ*(TileK/P) + (q+yq)*(TileK/P) + k/TileP+yk*VectorLen]);
    }}}
  } else {
    const uint32_t XTileSlices = TileK/P;
    const uint32_t XSlices     = K/P;

    for (uint32_t rm = 0; rm < VecRegM; rm++) {
    for (uint32_t rq = 0; rq < VecRegQ; rq++) {
    for (uint32_t rk = 0; rk < VecRegK; rk++) {
      auto reg = yReg[rm][rq][rk];
      const uint32_t cacheK = (rq + q) * XTileSlices + rk*VectorLen + k/TileP;
      if (fac > 0) {
        if (m + rm < XTile.m()) {
          // ElemT b[8]; _mm256_storeu_ps(b, reg); printf("%f %f %f %f\n", b[0], b[1], b[2], b[3]);
          reg.store(&tileBuff[(m+rm)*TileK + cacheK]);
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
          reg.store(Y.data<ElemT>(tileM + m + rm, memK, fastKronOp_N));
      }
    }}}
  }
}

template<typename ElemT, typename Vec2T, typename Vec4T,
         uint MaxQ, uint MaxP, uint TileP, uint TileQ, uint TileK,
         uint TileM, uint FusedFacs, uint RegK, uint RegQ,
         int XAlignment, int FAlignment, fastKronOp OpX, fastKronOp OpF>
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

  const uint32_t VectorLen = (XAlignment == 8 && RegK >= 8) ? 8 : 1; //AVX256 length
  
  // static_assert(XAlignment < 8 or (XAlignment == 8 and RegK % VectorLen == 0));
  static_assert(TileK % RegK == 0);
  static_assert(TileQ % RegQ == 0);
  assert(FusedFacs == 1 || (FusedFacs > 1 && P <= TileP && Q <= TileQ && P == Q));
  //For Transpose load loop to TileX
  assert ((TileK/P) % VectorLen == 0);

  uint threads = omp_get_max_threads();
  
  const size_t SzTileX = TileM*TileP*(TileK/P);
  //TODO: Allocate this in fastKron_initBackend
  static ElemT* TileXs[96] = {nullptr};
  static ElemT* TileYs[96] = {nullptr};
  static ElemT* TileFs[96] = {nullptr};

  if (TileXs[0] == nullptr) {
    for (int i = 0; i < 96; i++)  {
      TileXs[i] = (ElemT*)aligned_alloc(4096, SzTileX * sizeof(ElemT));
      TileYs[i] = (ElemT*)aligned_alloc(4096, TileM * TileQ * TileK/P * sizeof(ElemT));
      TileFs[i] = (ElemT*)aligned_alloc(4096, TileP * TileQ * sizeof(ElemT));
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
              FloatVectorType<VectorLen> slices[VectorLen];
              for (uint32_t sliceIdx = 0; sliceIdx < NumSlices; sliceIdx++) {
                const ElemT* ptr = (fac == FusedFacs - 1) ? XTile.data(m, k + sliceIdx*P + tileP + p, 0) :
                                                            &tileBuff[m * TileK + k + sliceIdx*P + tileP + p];
                slices[sliceIdx].load(ptr);
              }

              transpose<VectorLen>(slices);

              for (uint32_t pp = 0; pp < VectorLen; pp++) {
                slices[pp].store(&TileX[m*TileP*(TileK/P) + (p + pp)*(TileK/P) + k/P]);
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

        ElemT* TileF = TileFs[tid]; //[TileP][TileQ];
        Factor F = params.problem.f(fac);

        for (int p = 0; p < TileP; p++) {
          memcpy(&TileF[p*TileQ + 0], F.data<ElemT>(tileP + p, tileQ, OpF), TileQ * sizeof(ElemT));
        }
        
        for (uint32_t m = 0; m < XTile.m(); m += RegM) {
        for (uint32_t q = 0; q < TileQ; q += RegQ) {
        for (uint32_t k = 0; k < TileK/P * TileP; k += RegK * TileP) {
          vectorMMAAndStore<ElemT, VectorLen, MaxQ, MaxP, TileP, TileQ, TileK, TileM, FusedFacs, RegK, RegQ>(tileM, tileK, tileP, tileQ,
          m, q, k, fac, TileX, TileF, P, Q, K, XTile, tileBuff, Y, fusedParams);
        }}}
      }
    }
  }}}

  // free(TileXs);
}