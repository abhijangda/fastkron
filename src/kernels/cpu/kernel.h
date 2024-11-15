#include <cstring>
#include <cstdlib>
#include <omp.h>

#include "kernels/cpu/vector-types.h"
#include "kernels/cpu/tensor.h"
#include "kernels/get_batched_data.h"

enum EpilogueKind {
  None = 0,
  Alpha = 1 << 0,
  Beta = 1 << 1
};

#include "kernels/cpu/memory-store.h"
#include "kernels/cpu/cache-store.h"
#include "kernels/cpu/mma.h"

#pragma once

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

  for (int fac = ((FusedFacs == 1) ? 1 : params.problem.n()) - 1; fac >= 0; fac--) {
    TransposedDirectShared3D<ElemT, OpY, OptTileX, OptF, OptTileF> 
      TrXCache((ElemT*)params.caches->TileXs[tid]);

    bool isLastFactor = epilogueParams.isLastFactor && fac == 0;
    bool isFirstFactor = ((uint32_t)fac) == (((FusedFacs == 1) ? 1 : params.problem.n()) - 1);

    for (uint32_t tileP = 0; tileP < F.p(); tileP += OptTileF::P()) {
      DirectShared<OpF, ElemT, OptTileF::P(), OptTileF::Q()> FCache((ElemT*)params.caches->TileFs[tid]);

      F = F.sameShape(batchedData.getFBatch(params, fac, batch).data());
      //Transpose X data and store to TrXCache to reduce TLB misses
      transposeCache<OptLevel, EpilogueKindVal, ElemT, X86VecT, OpX, FusedFacs>(X, F, tileP, fac, isFirstFactor, isLastFactor, XTile, TrXCache, YCache, alphaVec, epilogueParams.template getAlpha<ElemT>());
      //Store F to FCache to reduce TLB misses
      directCache<OptLevel, ElemT, OpF>(F, FCache, tileP, tileQ);

      if (OpY == fastKronOp_N) {
        for (uint32_t m = 0; m < XTile.m()    ; m += YRegisters::m())   {
        for (uint32_t q = 0; q < OptTileF::Q(); q += YRegisters::q())   {
          const uint32_t TileSlices = (OptTileX::N()/OptF::P()) * OptTileF::P();
          const uint32_t SlicesIncr = YRegisters::k() * YRegisters::kvec() * OptTileF::P();
          for (uint32_t k = 0; k < TileSlices; k += SlicesIncr) {
            YRegisters YReg;
            YElem y(m, q, k);
            loadYInterim<X86VecT>(tileP, y, FCache, YCache, YReg);
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
          for (uint32_t m = 0; m < XTile.m() ; m += YRegisters::m() * YRegisters::mvec())   {
            YRegisters YReg;
            YElem y(m, q, k);

            loadYInterim<X86VecT>(tileP, y, FCache, YCache, YReg);
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
                          YRegisters<OpY, X86VecT, RegM, RegK, RegQ, 1, X86VecT::VectorLen>,
                          YRegisters<OpY, X86VecT, RegM, RegK, RegQ, X86VecT::VectorLen, 1>>::type;
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
    if (OpY == fastKronOp_N) {
      #pragma omp parallel for collapse(4)
      for (uint32_t batch = 0; batch < batchCount; batch++) {
      for (uint32_t tileQ = 0; tileQ < Q    ; tileQ += TileQ) {
      for (uint32_t tileM = 0; tileM < X.m(); tileM += TileM) {
      for (uint32_t tileK = 0; tileK < X.n(); tileK += TileK) {
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
    }}}}} else {
      #pragma omp parallel for collapse(4)
      for (uint32_t batch = 0; batch < batchCount; batch++) {
      for (uint32_t tileQ = 0; tileQ < Q    ; tileQ += TileQ) {
      for (uint32_t tileK = 0; tileK < X.n(); tileK += TileK) {
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
    }}}}
  }}
}