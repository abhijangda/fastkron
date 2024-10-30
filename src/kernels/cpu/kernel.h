#include <cstring>
#include <cstdlib>
#include <omp.h>

#include "kernels/cpu/vector-types.h"
#include "kernels/cpu/tensor.h"
#include "kernels/get_batched_data.h"

#pragma once

enum EpilogueKind {
  None = 0,
  Alpha = 1 << 0,
  Beta = 1 << 1
};

template<uint OptLevel, typename ElemT, fastKronOp OpF, typename DirectTileF>
static CUDA_DEVICE_HOST
void directCache(const Factor& F, DirectTileF& TileF, uint32_t tileP, uint32_t tileQ) {
  constexpr bool kQMultipleOfTileQ = KernelOptimizations::IsQMultipleOfTileQ(OptLevel);
  constexpr bool kPMultipleOfTileP = KernelOptimizations::IsPMultipleOfTileP(OptLevel);

  for (uint32_t row = 0; row < TileF.shape(0); row++) {
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

template<uint OptLevel, uint32_t EpilogueKindVal, typename ElemT, typename X86VecT, fastKronOp OpX,
         uint FusedFacs, typename TileX, typename XCache, typename YInterim>
static CUDA_DEVICE_HOST
void transposeCache(const Matrix& X, const Factor& F, uint32_t tileP, uint32_t fac, bool isLastFactor,
                    TileX& XTile, XCache& Xch, YInterim& Ych, X86VecT alphaVec, ElemT alpha) {
  const uint32_t VecTLen = X86VecT::VectorLen;
  const bool kPMultipleOfTileP = KernelOptimizations::IsPMultipleOfTileP(OptLevel);
  const bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);
  const bool kMMultipleOfTileM = KernelOptimizations::IsMMultipleOfTileM(OptLevel);
  const bool kTileKMultipleOfSlices = XTile.tileCols() % VecTLen == 0;

  if (Xch.layout() == fastKronOp_N) {
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
            if (fac == FusedFacs - 1 && 
                (EpilogueKindVal & EpilogueKind::Alpha) == EpilogueKind::Alpha)
              slices[pp].mul(alphaVec);
            slices[pp].store(&Xch.at(m, k/F.p(), p+pp));
          }
        } else {
          const uint32_t LeftSlices = (XTile.cols - k)/F.p();
          for (; p < MIN(Xch.p(), F.p() - tileP); p++) {
            for (uint32_t slice = 0; slice < LeftSlices; slice++) {
              const ElemT* ptr = (fac == FusedFacs - 1) ? 
                                  XTile.data(m, k/F.p() + slice, tileP + p) :
                                  &Ych.at(m,0,0) + k + slice*F.p() + tileP + p;
              ElemT val = *ptr;
              if (fac == FusedFacs - 1 &&
                  (EpilogueKindVal & EpilogueKind::Alpha) == EpilogueKind::Alpha) {
                val = alpha * val;
              }
              Xch.at(m, k/F.p() + slice, p) = val;
            }
          }

          Xch.zero(m,     k/F.p() + LeftSlices, p,
                  m + 1, k/F.p() + VecTLen,    Xch.p());
        }
      }
    }
  }
  } else if (Xch.layout() == fastKronOp_T) {
    for (uint32_t k = 0; k < XTile.cols; k += F.p()) {
    for (uint32_t p = 0; p < Xch.p(); p += VecTLen) {
    for (uint32_t m = 0; m < XTile.m(); m += VecTLen) {
      X86VecT slices[VecTLen];
      const bool UseAVXStore = 
          VecTLen > 1 && (kMMultipleOfTileM || XTile.m() - m >= VecTLen);

      if (UseAVXStore) {
        for (uint32_t pp = 0; pp < VecTLen; pp++) {
          const ElemT* ptr = (fac == 0) ? 
                              XTile.data(m, k/F.p(), tileP + p + pp) : 
                              &Ych.at(m,k/F.p(), tileP + p + pp);
          slices[pp].load(ptr);
        }

        for (uint32_t pp = 0; pp < VecTLen; pp++) {
          if (isLastFactor &&
              (EpilogueKindVal & EpilogueKind::Alpha) == EpilogueKind::Alpha) {
                slices[pp].mul(alphaVec);
          }
          slices[pp].store(&Xch.at(m, k/F.p(), p + pp));
        }
      } else {
        for (uint32_t pp = 0; pp < VecTLen; pp++) {
          const ElemT* ptr = (fac == 0) ? 
                              XTile.data(m, k/F.p(), tileP + p + pp) : 
                              &Ych.at(m,k/F.p(), tileP + p + pp);
          
          ElemT val = *ptr;

          if (isLastFactor &&
              (EpilogueKindVal & EpilogueKind::Alpha) == EpilogueKind::Alpha) {
            val = alpha * val;
          }

          Xch.at(m, k/F.p(), p + pp) = val;
        }
      }
  }}}
  }
}

template<typename X86VecT, 
         typename FCache, typename YInterim,
         typename YRegisters>
static CUDA_DEVICE_HOST
void load(uint32_t tileP, const YElem& y,
          const FCache& Fch, YInterim& Ych, YRegisters& YReg) {
  if (tileP == 0) {
    YReg.zero();
  } else {
    //TODO: For OpY=fastKronOp_T YReg.apply should have last loop in m
    const uint KVectorLen = 1; //(Ych.layout() == fastKronOp_N) ? X86VecT::VectorLen : 1;
    const uint MVectorLen = X86VecT::VectorLen;//(Ych.layout() == fastKronOp_N) ? 1 : X86VecT::VectorLen;
    YReg.apply(Ych.layout(), [&](X86VecT& e, const uint32_t ym, const uint32_t yk, const uint32_t yq) {
      e.load(&Ych.at(y.m() + ym * MVectorLen, y.q() + yq, y.k()/Fch.p() + yk * KVectorLen));
    });
  }
}

template<typename X86VecT, 
         typename XCache, typename FCache, typename YInterim,
         typename YRegisters>
static CUDA_DEVICE_HOST
void mma(uint32_t /*tileP*/, const YElem& y, 
         const XCache& Xch, const FCache& Fch,
         YInterim& Ych, YRegisters& YReg) {
  const uint VectorLen = X86VecT::VectorLen;

  for (uint32_t p = 0; p < Fch.p(); p++) {
    XRegisters<X86VecT, YReg.m(), YReg.k(), 1> XReg;
    FRegisters<X86VecT, 1, YReg.q()> FReg;
    XReg.apply(Ych.layout(), [&](X86VecT& e, const uint32_t em, const uint32_t ek, const uint32_t ep) {
      if (Ych.layout() == fastKronOp_N)
        e.load(&Xch.at(y.m() + em, y.k()/Fch.p() + ek*VectorLen, p + ep));
      else {
        e.load(&Xch.at(y.m() + em*VectorLen, y.k()/Fch.p() + ek, p + ep));
      }
    });

    FReg.apply([&](X86VecT& e, const uint32_t ep, const uint32_t eq) {
      e.broadcast(&Fch.at(p + ep, y.q() + eq));
    });

    YReg.apply(Ych.layout(), [&](X86VecT& e, const uint32_t ym, const uint32_t yk, const uint32_t yq) {
      e.fmadd(XReg.at(ym, yk, 0), FReg.at(0, yq));
    });
  }
}

template<uint OptLevel, uint32_t EpilogueKindVal,
         typename ElemT, typename X86VecT,
         typename KernelParams, typename FusedParams, typename EpilogueParams,
         typename TileX, typename FCache, typename YInterim, typename YRegisters>
static CUDA_DEVICE_HOST
void store(const KernelParams& /*params*/, const FusedParams& fusedParams, const EpilogueParams& /*epilogueParams*/, 
           X86VecT beta,
           uint32_t fac, uint32_t /*batch*/,
           uint32_t tileM, uint32_t tileK, uint32_t tileP, uint32_t tileQ,
           const YElem& y, 
           const Factor& F, Matrix& Y, Matrix& Z, FCache& Fch, TileX& XTile,
           YInterim& Ych, YRegisters& YReg) {
  const uint KVectorLen = 1;//X86VecT::VectorLen;// (Ych.layout() == fastKronOp_N) ? X86VecT::VectorLen : 1;
  const uint MVectorLen = X86VecT::VectorLen;//1;//X86VecT::VectorLen;//(Ych.layout() == fastKronOp_N) ? 1 : X86VecT::VectorLen;

  if ((Ych.layout() == fastKronOp_N && fac > 0) ||
      (Ych.layout() == fastKronOp_T && fac < fusedParams.NumFused - 1) ||
      (Fch.p() <= F.p() && tileP < F.p() - Fch.p())) {
    YReg.apply(Ych.layout(), [&](X86VecT& e, const uint32_t rm, const uint32_t rk, const uint32_t rq) {
      e.store(&Ych.at(y.m()+rm*MVectorLen, y.q() + rq, y.k()/Fch.p() + rk * KVectorLen));
    });
  } else {
    YReg.apply(Ych.layout(), [&](X86VecT& e, const uint32_t rm, const uint32_t rk, const uint32_t rq) {
      constexpr bool kQMultipleOfTileQ = KernelOptimizations::IsQMultipleOfTileQ(OptLevel);
      constexpr bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);
      constexpr bool kMMultipleOfTileM = KernelOptimizations::IsMMultipleOfTileM(OptLevel);

      uint32_t slice = y.k()/Fch.p() + rk * KVectorLen;

      if (!kKMultipleOfTileK && slice >= XTile.cols/F.p()) return;
      if (!kQMultipleOfTileQ && tileQ + y.q() + rq >= F.q()) return;

      const uint32_t XTileSlices = XTile.tileCols()/F.p();
      const uint32_t XSlices     = Y.n()/F.q();
      uint32_t yN;

      if (fusedParams.NumFused > 1) {
        uint32_t xshCol = (rq + y.q()) * XTileSlices + rk*KVectorLen + y.k()/Fch.p();
        //Scale shared mem slice idx to global mem idx
        uint32_t glSlice = (xshCol/XTileSlices)*XSlices;
        //Scale shared fused slice to global mem
        uint32_t sliceElem = ((xshCol%XTileSlices)/fusedParams.XShFusedSlices)*fusedParams.XglFusedSlices;
        //Elem idx in Fused Slice
        uint32_t elem = (tileK/XTile.tileCols()) * fusedParams.XShFusedSlices +
                        xshCol%fusedParams.XShFusedSlices;
        yN = glSlice + sliceElem + elem; 
      } else {
        yN = (y.q() + rq) * XSlices +
             (tileK/XTile.tileCols()) * XTileSlices +
             slice;
        if (Fch.q() < F.q()) {
          yN += tileQ * XSlices;
        }
      }

      if (kMMultipleOfTileM || y.m() + rm*MVectorLen < XTile.m()) {
        uint32_t numElems;
        if (Ych.layout() == fastKronOp_N) {
          uint32_t slices = (kKMultipleOfTileK &&
                            XTile.tileCols() % KVectorLen == 0) ? 
                            KVectorLen : (XTile.cols/F.p() - slice);
          slices = MIN(KVectorLen, slices);
          numElems = slices;
        } else {
          numElems = kMMultipleOfTileM ? MVectorLen : XTile.m() - (y.m() + rm*MVectorLen);
          numElems = MIN(MVectorLen, numElems);
        }
        if ((EpilogueKindVal & EpilogueKind::Beta) == EpilogueKind::Beta) {
          X86VecT z;
          z.load(Z.data<ElemT>((tileM + y.m() + rm), yN, Ych.layout()), numElems);
          e.fmadd(beta, z);
        }
        e.store(Y.data<ElemT>(tileM + y.m() + rm*MVectorLen, yN, Ych.layout()), numElems);
    }});
  }
}

template<typename ElemT, typename X86VecT, 
         fastKronOp OpX, fastKronOp OpF, fastKronOp OpY,
         uint OptLevel, uint32_t EpilogueKindVal, uint FusedFacs,
         typename OptF, typename OptTileF, typename OptTileX,
         KernelBatchType::Ty KernelBatch,
         typename YRegisters, typename KernelParams, typename FusedParams, typename EpilogueParams>
void threadWork(KernelParams& params,
                FusedParams& fusedParams,
                EpilogueParams& epilogueParams,
                uint32_t batch, uint32_t tileM, uint32_t tileK, uint32_t tileQ, uint32_t TileK) {
  // constexpr bool kXshSlicesSame    = KernelOptimizations::IsXshSlicesSame   (OptLevel);
  constexpr bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);
  constexpr bool kTileKSame        = KernelOptimizations::IsTileKSame       (OptLevel);
  constexpr bool kFactorShapeSame  = KernelOptimizations::IsFactorShapeSame (OptLevel);
  GetBatchedData<KernelBatch, ElemT, KernelParams, EpilogueParams> batchedData;

  Matrix X = batchedData.getXBatch(params, batch);
  Matrix Y = batchedData.getYBatch(params, batch);
  Factor F = (kFactorShapeSame) ? Factor(OptF::P(), OptF::Q(), batchedData.getFBatch(params, 0, batch).data()) :
                                  batchedData.getFBatch(params, 0, batch);
  Matrix Z = batchedData.getZBatch(epilogueParams, Y, batch);

  SliceCPU<ElemT, kKMultipleOfTileK, kTileKSame, OptTileX> XTile(tileM, tileK, TileK, F.p(), X);

  const uint tid = omp_get_thread_num();

  YInterim<ElemT, OpY, OptTileX, OptTileF, OptF> YCache((ElemT*)params.caches->TileYs[tid]);
  X86VecT alphaVec;
  X86VecT betaVec;

  if ((EpilogueKindVal & EpilogueKind::Alpha) == EpilogueKind::Alpha) {
    ElemT alpha = epilogueParams.template getAlpha<ElemT>();
    alphaVec.broadcast(&alpha);
  }

  if ((EpilogueKindVal & EpilogueKind::Beta) == EpilogueKind::Beta) {
    ElemT beta = epilogueParams.template getBeta<ElemT>();
    betaVec.broadcast(&beta);
  }

  for (int _fac = FusedFacs - 1; _fac >= 0; _fac--) {
    TransposedDirectShared3D<ElemT, OpY, OptTileX, OptF, OptTileF> 
      TrXCache((ElemT*)params.caches->TileXs[tid]);

    int fac = 0;
    if (OpY == fastKronOp_N)      fac = _fac;
    else if (OpY == fastKronOp_T) fac = FusedFacs - 1 - _fac;

    for (uint32_t tileP = 0; tileP < F.p(); tileP += OptTileF::P()) {
      DirectShared<OpF, ElemT, OptTileF::P(), OptTileF::Q()> FCache((ElemT*)params.caches->TileFs[tid]);

      F = F.sameShape(batchedData.getFBatch(params, fac, batch).data());
      //Transpose X data and store to TrXCache to reduce TLB misses
      transposeCache<OptLevel, EpilogueKindVal, ElemT, X86VecT, OpX, FusedFacs>(X, F, tileP, fac, epilogueParams.isLastFactor, XTile, TrXCache, YCache, alphaVec, epilogueParams.template getAlpha<ElemT>());
      //Store F to FCache to reduce TLB misses
      directCache<OptLevel, ElemT, OpF>(F, FCache, tileP, tileQ);

      if (OpY == fastKronOp_N) {
        for (uint32_t m = 0; m < XTile.m()    ; m += YRegisters::m())   {
        for (uint32_t q = 0; q < OptTileF::Q(); q += YRegisters::q())   {
          const uint32_t TileSlices = (OptTileX::N()/OptF::P()) * OptTileF::P();
          const uint32_t SlicesIncr = YRegisters::k() * X86VecT::VectorLen * OptTileF::P();
          for (uint32_t k = 0; k < TileSlices; k += SlicesIncr) {
            YRegisters YReg;
            YElem y(m, q, k);
            load<X86VecT>(tileP, y, FCache, YCache, YReg);
            mma<X86VecT>(tileP, y, TrXCache, FCache, YCache, YReg);
            store<OptLevel, EpilogueKindVal, ElemT, X86VecT>(params, fusedParams, epilogueParams, betaVec,
                                                             fac, batch, tileM, tileK, tileP, tileQ,
                                                             y, F, Y, Z, FCache, XTile, YCache, YReg);
        }}}
      } else if (OpY == fastKronOp_T) {
        for (uint32_t q = 0; q < OptTileF::Q(); q += YRegisters::q())   {
          const uint32_t TileSlices = (OptTileX::N()/OptF::P()) * OptTileF::P();
          const uint32_t SlicesIncr = YRegisters::k() * OptTileF::P();
          for (uint32_t k = 0; k < TileSlices; k += SlicesIncr) {
          for (uint32_t m = 0; m < XTile.m() ; m += YRegisters::m() * X86VecT::VectorLen)   {
            YRegisters YReg;
            YElem y(m, q, k);

            load<X86VecT>(tileP, y, FCache, YCache, YReg);
            mma<X86VecT>(tileP, y, TrXCache, FCache, YCache, YReg);
            store<OptLevel, EpilogueKindVal, ElemT, X86VecT>(params, fusedParams, epilogueParams, betaVec,
                                                             fac, batch, tileM, tileK, tileP, tileQ,
                                                             y, F, Y, Z, FCache, XTile, YCache, YReg);
        }}}
      }
    }
  }
}

template<typename ElemT, typename X86VecT, uint MaxP, uint MaxQ, uint TileP, 
         uint TileQ, uint kTileK, uint TileM, uint FusedFacs, 
         uint RegM, uint RegK, uint RegQ, uint OptLevel, 
         int XAlignment, int FAlignment,
         fastKronOp kOpX, fastKronOp kOpF, FastKronMMType MMType,
         KernelBatchType::Ty KernelBatch,
         typename KernelParams, typename FusedParams, typename EpilogueParams>
void cpuKernel(KernelParams& params,
               FusedParams& fusedParams,
               DistributedParams& /*distParams*/,
               EpilogueParams& epilogueParams) {
  const fastKronOp OpX = (MMType == FastKronMMType::MKM) ? kOpX : swapFastKronOp<kOpX>();
  const fastKronOp OpF = (MMType == FastKronMMType::MKM) ? kOpF : swapFastKronOp<kOpF>();
  const fastKronOp OpY = (MMType == FastKronMMType::MKM) ? fastKronOp_N : fastKronOp_T;

  using OptF  = FixedShapeFactor<fastKronOp_N, ElemT, MaxP, MaxQ>;
  using OptTileF = FixedShapeFactor<OpF, ElemT, TileP, TileQ>;
  //TODO: YRegisters should have VectorLen for both M and K.
  //TODO: Instead of VectorLen use MVectorLen or KVectorLen in code
  using YRegs = typename std::conditional<MMType == FastKronMMType::MKM, 
                          YRegisters<X86VecT, RegM, RegK/X86VecT::VectorLen, RegQ>,
                          YRegisters<X86VecT, RegM/X86VecT::VectorLen, RegK, RegQ>>::type;
  using OptTileX = FixedShapeMatrix<OpX, ElemT, TileM, kTileK>;

  static_assert(TileM % RegM == 0);
  static_assert(kTileK % RegK == 0);
  static_assert(TileQ % RegQ == 0);
  static_assert(FusedFacs == 1 ||
                (FusedFacs > 1 &&
                 MaxP <= TileP && MaxQ <= TileQ && MaxP == MaxQ &&
                 OptLevel == KernelOptimizations::MaxOptLevel()));

  constexpr bool kFactorShapeSame = KernelOptimizations::IsFactorShapeSame(OptLevel);

  Matrix X = params.problem.x();
  // Matrix Y = params.problem.y();
  Factor F = params.problem.f(0);

  const uint Q = (kFactorShapeSame) ? MaxQ : F.q();
  // const uint P = (kFactorShapeSame) ? MaxP : F.p();

  // const uint XshSlices = getXshSlices<OptLevel, kTileK, MaxP>(params);
  // const uint XSlices   = getXSlices  <OptLevel, MaxQ>(Y, params);
  const uint TileK     = getXTileK   <OptLevel, kTileK>(params);
  const bool hasAlpha  = epilogueParams.template getAlpha<ElemT>() != (ElemT)1.0f;
  const bool hasBeta   = epilogueParams.template getD<ElemT>() != nullptr && 
                         epilogueParams.template getBeta<ElemT>() != (ElemT)0;
  const bool notLastFactor = not epilogueParams.isLastFactor;

  const uint32_t batchCount = GetBatchedData<KernelBatch, ElemT, KernelParams, EpilogueParams>().getBatchCount(params);

  if (OpX == fastKronOp_N) {
    #pragma omp parallel for collapse(4)
    for (uint32_t batch = 0; batch < batchCount; batch++)
    for (uint32_t tileM = 0; tileM < X.m(); tileM += TileM) {
    for (uint32_t tileK = 0; tileK < X.n(); tileK += TileK) {
    for (uint32_t tileQ = 0; tileQ < Q    ; tileQ += TileQ) {
      if (notLastFactor || (!hasAlpha && !hasBeta)) {
        threadWork<ElemT, X86VecT, OpX, OpF, OpY, OptLevel, EpilogueKind::None, FusedFacs, OptF, OptTileF, OptTileX, KernelBatch, YRegs> (
          params, fusedParams, epilogueParams, batch, tileM, tileK, tileQ, TileK
        );
      } else if (hasAlpha && !hasBeta) {
        threadWork<ElemT, X86VecT, OpX, OpF, OpY, OptLevel, EpilogueKind::Alpha, FusedFacs, OptF, OptTileF, OptTileX, KernelBatch, YRegs> (
            params, fusedParams, epilogueParams, batch, tileM, tileK, tileQ, TileK
        );
      } else if (hasBeta) {
        threadWork<ElemT, X86VecT, OpX, OpF, OpY, OptLevel, EpilogueKind::Alpha | EpilogueKind::Beta, FusedFacs, OptF, OptTileF, OptTileX, KernelBatch, YRegs> (
            params, fusedParams, epilogueParams, batch, tileM, tileK, tileQ, TileK
        );
      }
    }}}
  } else if (OpX == fastKronOp_T) {
    #pragma omp parallel for collapse(4)
    for (uint32_t batch = 0; batch < batchCount; batch++)
    for (uint32_t tileQ = 0; tileQ < Q    ; tileQ += TileQ) {
    for (uint32_t tileK = 0; tileK < X.n(); tileK += TileK) { //TODO: swap X.n() and X.m() for Opx = fastKronOpT and MKM 
    for (uint32_t tileM = 0; tileM < X.m(); tileM += TileM) {
      if (notLastFactor || (!hasAlpha && !hasBeta)) {
        threadWork<ElemT, X86VecT, OpX, OpF, OpY, OptLevel, EpilogueKind::None, FusedFacs, OptF, OptTileF, OptTileX, KernelBatch, YRegs> (
          params, fusedParams, epilogueParams, batch, tileM, tileK, tileQ, TileK
        );
      }
      else if (hasAlpha && !hasBeta) {
        threadWork<ElemT, X86VecT, OpX, OpF, OpY, OptLevel, EpilogueKind::Alpha, FusedFacs, OptF, OptTileF, OptTileX, KernelBatch, YRegs> (
            params, fusedParams, epilogueParams, batch, tileM, tileK, tileQ, TileK
        );
      } else if (hasBeta) {
        threadWork<ElemT, X86VecT, OpX, OpF, OpY, OptLevel, EpilogueKind::Alpha | EpilogueKind::Beta, FusedFacs, OptF, OptTileF, OptTileX, KernelBatch, YRegs> (
            params, fusedParams, epilogueParams, batch, tileM, tileK, tileQ, TileK
        );
      }
    }}}
  }
}