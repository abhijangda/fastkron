#include <cstring>
#include <cstdlib>
#include <omp.h>

#include "kernels/cpu/vector-types.h"
#include "kernels/cpu/tensor.h"

#pragma once

template<uint OptLevel, typename ElemT, fastKronOp OpF, typename DirectTileF>
static CUDA_DEVICE_HOST inline
void directCache(const Factor& F, DirectTileF& TileF, uint32_t tileP, uint32_t tileQ) {
  constexpr bool kQMultipleOfTileQ = KernelOptimizations::IsQMultipleOfTileQ(OptLevel);
  constexpr bool kPMultipleOfTileP = KernelOptimizations::IsPMultipleOfTileP(OptLevel);

  for (int row = 0; row < TileF.shape(0); row++) {
    if ((OpF == fastKronOp_N && (kPMultipleOfTileP || tileP + row < F.p())) ||
        (OpF == fastKronOp_T && (kQMultipleOfTileQ || tileQ + row < F.q()))) {
      uint32_t row_elems;
      ElemT* Fptr;
      if (OpF == fastKronOp_N) {
        row_elems = kQMultipleOfTileQ ? TileF.q() : MIN(TileF.q(), F.q() - tileQ);
        Fptr = F.data<ElemT>(tileP + row, tileQ, OpF);
      } else if (OpF == fastKronOp_T) {
        row_elems = kPMultipleOfTileP ? TileF.p() : MIN(TileF.p(), F.p() - tileP);
        Fptr = F.data<ElemT>(tileP, tileQ + row, OpF);
      }

      TileF.store_row(row, row_elems, Fptr);
    } else {
      TileF.zero_row(row);
    } 
  }
}

template<uint OptLevel, typename ElemT, typename X86VecT, fastKronOp OpX,
         uint FusedFacs, typename TileX, typename XCache, typename YInterim>
static CUDA_DEVICE_HOST inline
void transposeCache(const Matrix& X, const Factor& F, uint32_t tileP, int fac,
                    TileX& XTile, XCache& Xch, YInterim& Ych) {
  const uint32_t VecTLen = X86VecT::VectorLen;
  const bool kPMultipleOfTileP = KernelOptimizations::IsPMultipleOfTileP(OptLevel);
  const bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);
  const bool kTileKMultipleOfSlices = XTile.tileCols() % VecTLen == 0;

  for (uint32_t m = 0; m < XTile.m(); m++) {
    for (uint32_t k = 0; k < XTile.cols; k += VecTLen * F.p()) {
      uint32_t p = 0;
      for (; p < Xch.p(); p += VecTLen) {
        const bool UseAVXTrans = 
          VecTLen > 1 &&
          ((kKMultipleOfTileK && kTileKMultipleOfSlices) || XTile.cols - k    >= VecTLen * F.p()) && 
          ((kPMultipleOfTileP && Xch.p() % VecTLen == 0) || F.p() - tileP - p >= VecTLen) &&
          (Xch.p() >= VecTLen);
        if (UseAVXTrans) {
          X86VecT slices[VecTLen];
          if (OpX == fastKronOp_N || (OpX == fastKronOp_T and fac < FusedFacs - 1)) {
            for (uint32_t slice = 0; slice < VecTLen; slice++) {
              const ElemT* ptr = (fac == FusedFacs - 1) ? 
                                 XTile.data(m, k/F.p() + slice, tileP + p) :
                                 &Ych.at(m,0,0) + k + slice*F.p() + tileP + p;
              slices[slice].load(ptr);
            }
            X86VecT::transpose(slices);
          } else if (OpX == fastKronOp_T and fac == FusedFacs - 1) {
            //Gather requires AVX2
            uint32_t gatherIdxs[VecTLen] = {0};
            for (uint pp = 0; pp < VecTLen; pp++) {
              const ElemT* ptr = XTile.data(m, k/F.p() + 0, tileP + p + pp);
              for (uint32_t slice = 0; slice < VecTLen; slice++) {
                gatherIdxs[slice] = slice * X.m() * F.p(); //TODO: Assumes TileM == 1
              }

              slices[pp].gather(ptr, gatherIdxs);
            }
          }

          for (uint32_t pp = 0; pp < VecTLen; pp++) {
            slices[pp].store(&Xch.at(m, k/F.p(), p+pp));
          }
        } else {
          const uint32_t LeftSlices = (XTile.cols - k)/F.p();
          for (; p < MIN(Xch.p(), F.p() - tileP); p++) {
            for (uint32_t slice = 0; slice < LeftSlices; slice++) {
              const ElemT* ptr = (fac == FusedFacs - 1) ? 
                                  XTile.data(m, k/F.p() + slice, tileP + p) :
                                  &Ych.at(m,0,0) + k + slice*F.p() + tileP + p;
              Xch.at(m, k/F.p() + slice, p) = *ptr;
            }
          }

          Xch.zero(m,     k/F.p() + LeftSlices, p,
                   m + 1, k/F.p() + VecTLen,    Xch.p());
        }
      }
    }
  }
}

template<typename X86VecT, 
         typename FCache, typename YInterim,
         typename YRegisters>
static CUDA_DEVICE_HOST inline
void load(uint32_t tileP, uint32_t m, uint32_t q, uint32_t k, 
          const FCache& Fch, YInterim& Ych, YRegisters& YReg) {
  const uint VectorLen = X86VecT::VectorLen;
  if (tileP == 0) {
    YReg.zero();
  } else {
    YReg.apply([&](X86VecT& e, const uint32_t ym, const uint32_t yk, const uint32_t yq) {
      e.load(&Ych.at(m + ym, q + yq, k/Fch.p() + yk * VectorLen));
    });
  }
}

template<typename X86VecT, 
         typename XCache, typename FCache, typename YInterim,
         typename YRegisters>
static CUDA_DEVICE_HOST inline
void mma(uint32_t tileP, uint32_t m, uint32_t q, uint32_t k, 
         const XCache& Xch, const FCache& Fch,
         YInterim& Ych, YRegisters& YReg) {
  const uint VectorLen = X86VecT::VectorLen;

  for (uint32_t p = 0; p < Fch.p(); p++) {
    XRegisters<X86VecT, YReg.m(), YReg.k(), 1> XReg;
    FRegisters<X86VecT, 1, YReg.q()> FReg;
    XReg.apply([&](X86VecT& e, const uint32_t em, const uint32_t ek, const uint32_t ep) {
    e.load(&Xch.at((m + em), k/Fch.p() + ek*VectorLen, p + ep));
    });

    FReg.apply([&](X86VecT& e, const uint32_t ep, const uint32_t eq) {
    e.broadcast(&Fch.at(p + ep, q + eq));
    });

    YReg.apply([&](X86VecT& e, const uint32_t ym, const uint32_t yk, const uint32_t yq) {
    e.fmadd(XReg.at(ym, yk, 0), FReg.at(0, yq));
    });
  } 
}

template<uint OptLevel,
         typename ElemT, typename X86VecT,
         typename FusedParams,
         typename TileX, typename FCache, typename YInterim, typename YRegisters>
static CUDA_DEVICE_HOST inline
void store(const FusedParams& fusedParams, uint32_t fac,
           uint32_t tileM, uint32_t tileK, uint32_t tileP, uint32_t tileQ,
           uint32_t m, uint32_t q, uint32_t k, 
           const Factor& F, Matrix& Y, FCache& Fch, TileX& XTile,
           YInterim& Ych, YRegisters& YReg) {
  const uint VectorLen = X86VecT::VectorLen;

  if (fac > 0 || (Fch.p() <= F.p() && tileP < F.p() - Fch.p())) {
    YReg.apply([&](X86VecT& e, const uint32_t rm, const uint32_t rk, const uint32_t rq) {
      e.store(&Ych.at(m+rm, q + rq, k/Fch.p() + rk * VectorLen));
    });
  } else {
    YReg.apply([&](X86VecT& e, const uint32_t rm, const uint32_t rk, const uint32_t rq) {
      constexpr bool kQMultipleOfTileQ = KernelOptimizations::IsQMultipleOfTileQ(OptLevel);
      constexpr bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);
      uint32_t slice = k/Fch.p() + rk*VectorLen;

      if (!kKMultipleOfTileK && slice >= XTile.cols/F.p()) return;
      if (!kQMultipleOfTileQ && tileQ + q + rq >= F.q()) return;

      const uint32_t XTileSlices = XTile.tileCols()/F.p();
      const uint32_t XSlices     = Y.n()/F.q();
      uint32_t yN;

      if (fusedParams.NumFused > 1) {
        uint32_t xshCol = (rq + q) * XTileSlices + rk*VectorLen + k/Fch.p();
        //Scale shared mem slice idx to global mem idx
        uint32_t glSlice = (xshCol/XTileSlices)*XSlices;
        //Scale shared fused slice to global mem
        uint32_t sliceElem = ((xshCol%XTileSlices)/fusedParams.XShFusedSlices)*fusedParams.XglFusedSlices;
        //Elem idx in Fused Slice
        uint32_t elem = (tileK/XTile.tileCols()) * fusedParams.XShFusedSlices +
                        xshCol%fusedParams.XShFusedSlices;
        yN = glSlice + sliceElem + elem; 
      } else {
        yN = (q + rq) * XSlices +
                (tileK/XTile.tileCols()) * XTileSlices +
                slice;
        if (Fch.q() < F.q()) {
          yN += tileQ * XSlices;
        }
      }

      if (m + rm < XTile.m()) {
        uint32_t slices = (kKMultipleOfTileK &&
                           XTile.tileCols() % VectorLen == 0) ? 
                           VectorLen : (XTile.cols/F.p() - slice);
        slices = MIN(VectorLen, slices);
        e.store(Y.data<ElemT>(tileM + m + rm, yN, fastKronOp_N), slices);
    }});
  }
}

template<typename ElemT, typename X86VecT, 
         fastKronOp OpX, fastKronOp OpF,
         uint OptLevel, uint FusedFacs,
         typename OptF, typename OptTileF, typename OptTileX,
         typename YRegisters>
void threadWork(KernelParams<FusedFacs>& params,
               FusedParams<FusedFacs>& fusedParams, 
               uint32_t tileM, uint32_t tileK, uint32_t tileQ, uint32_t TileK) {
  constexpr bool kXshSlicesSame    = KernelOptimizations::IsXshSlicesSame   (OptLevel);
  constexpr bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);
  constexpr bool kTileKSame        = KernelOptimizations::IsTileKSame       (OptLevel);
  constexpr bool kFactorShapeSame  = KernelOptimizations::IsFactorShapeSame (OptLevel);

  Matrix X = params.problem.x();
  Matrix Y = params.problem.y();
  Factor F = (kFactorShapeSame) ? Factor(OptF::P(), OptF::Q(), params.problem.f(0).data()) :
                                  params.problem.f(0);

  SliceCPU<ElemT, OpX, kKMultipleOfTileK, kTileKSame, OptTileX::M(), OptTileX::N()> XTile(tileM, tileK, TileK, F.p(), X);

  const uint tid = omp_get_thread_num();
  YInterim<ElemT, OptTileX::M(), OptTileF::Q(), OptTileX::Slices()> YCache((ElemT*)params.TileYs[tid]);

  for (int fac = FusedFacs - 1; fac >= 0; fac--) {
    TransposedDirectShared3D<fastKronOp_N, ElemT, OptTileX::M(), OptTileF::P(), OptTileX::Slices()> 
      TrXCache((ElemT*)params.TileXs[tid]);

    //Transpose X data and store to TrXCache to reduce TLB misses
    for (uint32_t tileP = 0; tileP < F.p(); tileP += OptTileF::P()) {
      F = F.shapeLike(params.problem.f(fac).data());

      transposeCache<OptLevel, ElemT, X86VecT, OpX, FusedFacs>(X, F, tileP, fac, XTile, TrXCache, YCache);

      DirectShared<OpF, ElemT, OptTileF::P(), OptTileF::Q()> FCache((ElemT*)params.TileFs[tid]);
      directCache<OptLevel, ElemT, OpF>(F, FCache, tileP, tileQ);

      for (uint32_t m = 0; m < XTile.m();     m += YRegisters::m())   {
      for (uint32_t q = 0; q < OptTileF::Q(); q += YRegisters::q())   {
      for (uint32_t k = 0; k < OptTileX::Slices() * OptTileF::P();
           k += YRegisters::k() * X86VecT::VectorLen * OptTileF::P()) {
        YRegisters YReg;

        load<X86VecT>(tileP, m, q, k, FCache, YCache, YReg);
        mma<X86VecT>(tileP, m, q, k, TrXCache, FCache, YCache, YReg);
        store<OptLevel, ElemT, X86VecT>(fusedParams, fac, tileM, tileK, tileP, tileQ,
                                        m, q, k, F, Y, FCache, XTile, YCache, YReg);
      }}}
    }
  }
}

template<typename ElemT, typename X86VecT, uint MaxP, uint MaxQ, uint TileP, 
         uint TileQ, uint kTileK, uint TileM, uint FusedFacs, 
         uint RegM, uint RegK, uint RegQ, uint OptLevel, 
         int XAlignment, int FAlignment,
         fastKronOp OpX, fastKronOp OpF>
void cpuKernel(KernelParams<FusedFacs>& params,
               FusedParams<FusedFacs>& fusedParams,
               DistributedParams& distParams,
               EpilogueParams& epilogueParams) {
  using OptF  = FixedShapeFactor<ElemT, MaxP, MaxQ>;
  using OptTileF = FixedShapeFactor<ElemT, TileP, TileQ>;
  using YRegs = YRegisters<X86VecT, RegM, RegK/X86VecT::VectorLen, RegQ>;
  using OptTileX = FixedShapeMatrix<ElemT, TileM, kTileK, MaxP>;

  static_assert(RegM == TileM, "x86 requires RegM == TileM");

  static_assert(kTileK % RegK == 0);
  static_assert(TileQ % RegQ == 0);
  static_assert(FusedFacs == 1 || 
                (FusedFacs > 1 && 
                 MaxP <= TileP && MaxQ <= TileQ && MaxP == MaxQ && 
                 OptLevel == KernelOptimizations::MaxOptLevel()));

  constexpr bool kFactorShapeSame = KernelOptimizations::IsFactorShapeSame(OptLevel);

  Matrix X = params.problem.x();
  Matrix Y = params.problem.y();
  Factor F = params.problem.f(0);

  const uint Q = (kFactorShapeSame) ? MaxQ : F.q();
  const uint P = (kFactorShapeSame) ? MaxP : F.p();

  const uint XshSlices = getXshSlices<OptLevel, kTileK, MaxP>(params);
  const uint XSlices   = getXSlices  <OptLevel, MaxQ>(Y, params);
  const uint TileK     = getXTileK   <OptLevel, kTileK>(params);

  if (OpX == fastKronOp_N) {
    #pragma omp parallel for collapse(3)
    for (uint32_t tileM = 0; tileM < X.m(); tileM += TileM) {
    for (uint32_t tileK = 0; tileK < X.n(); tileK += TileK) {
    for (uint32_t tileQ = 0; tileQ < Q    ; tileQ += TileQ) {
      threadWork<ElemT, X86VecT, OpX, OpF, OptLevel, FusedFacs, OptF, OptTileF, OptTileX, YRegs> (
        params, fusedParams, tileM, tileK, tileQ, TileK
      );
    }}}
  } else if (OpX == fastKronOp_T) {
    #pragma omp parallel for collapse(3)
    for (uint32_t tileQ = 0; tileQ < Q    ; tileQ += TileQ) {
    for (uint32_t tileM = 0; tileM < X.m(); tileM += TileM) {
    for (uint32_t tileK = 0; tileK < X.n(); tileK += TileK) {
      threadWork<ElemT, X86VecT, OpX, OpF, OptLevel, FusedFacs, OptF, OptTileF, OptTileX, YRegs> (
        params, fusedParams, tileM, tileK, tileQ, TileK
      );
    }}}
  }
}