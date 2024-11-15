#pragma once

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
void transposeCache(const Matrix& X, const Factor& F, uint32_t tileP, uint32_t /*fac*/, bool isFirstFactor, bool isLastFactor,
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
          ((kKMultipleOfTileK && kTileKMultipleOfSlices) || XTile.cols >= VecTLen * F.p() + k) && 
          ((kPMultipleOfTileP && Xch.p() % VecTLen == 0) || F.p() >= VecTLen + tileP + p) &&
          (Xch.p() >= VecTLen);
        if (UseAVXTrans) {
          X86VecT slices[VecTLen];
          if (OpX == fastKronOp_N || (OpX == fastKronOp_T and !isFirstFactor)) {
            for (uint32_t slice = 0; slice < VecTLen; slice++) {
              const ElemT* ptr = (isFirstFactor) ?
                                XTile.data(m, k/F.p() + slice, tileP + p) :
                                &Ych.at(m,0,0) + k + slice*F.p() + tileP + p;
              slices[slice].load(ptr);
            }
            X86VecT::transpose(slices);
          } else if (OpX == fastKronOp_T and isFirstFactor) {
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
            if (isFirstFactor && 
                (EpilogueKindVal & EpilogueKind::Alpha) == EpilogueKind::Alpha)
              slices[pp].mul(alphaVec);
            slices[pp].store(&Xch.at(m, k/F.p(), p+pp));
          }
        } else {
          const uint32_t LeftSlices = (XTile.cols - k)/F.p();
          for (; p < MIN(Xch.p(), F.p() - tileP); p++) {
            for (uint32_t slice = 0; slice < LeftSlices; slice++) {
              const ElemT* ptr = (isFirstFactor) ? 
                                  XTile.data(m, k/F.p() + slice, tileP + p) :
                                  &Ych.at(m,0,0) + k + slice*F.p() + tileP + p;
              ElemT val = *ptr;
              if (isFirstFactor &&
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
    uint32_t p = 0;
    for (; p < Xch.p(); p += VecTLen) {
    uint32_t m = 0;
    for (; m < XTile.m(); m += VecTLen) {
      X86VecT slices[VecTLen];
      const bool UseAVXStore = 
          VecTLen > 1 && 
          (kMMultipleOfTileM || XTile.m() - m >= VecTLen) &&
          ((kPMultipleOfTileP && Xch.p() % VecTLen == 0) || (F.p() >= VecTLen + tileP + p));

      if (UseAVXStore) {
        if (OpX == fastKronOp_T || !isFirstFactor) {
          for (uint32_t pp = 0; pp < VecTLen; pp++) {
            const ElemT* ptr = (isFirstFactor) ? 
                                XTile.data(m, k/F.p(), tileP + p + pp) : 
                                &Ych.at(0, 0, 0) + (k + p + pp)*Ych.m() + m;
            slices[pp].load(ptr);
          }
        } else if (OpX == fastKronOp_N) {
          for (uint32_t mm = 0; mm < VecTLen; mm++) {
            const ElemT* ptr = XTile.data(m+mm, k/F.p(), tileP + p);
            slices[mm].load(ptr);
          }

          X86VecT::transpose(slices);
        }

        for (uint32_t pp = 0; pp < VecTLen; pp++) {
          if (isLastFactor &&
              (EpilogueKindVal & EpilogueKind::Alpha) == EpilogueKind::Alpha) {
                slices[pp].mul(alphaVec);
          }
          slices[pp].store(&Xch.at(m, k/F.p(), p + pp));
        }
      } else if (XTile.m() - m < VecTLen && F.p() > tileP + p) {
        for (uint32_t pp = 0; pp < MIN(VecTLen, F.p() - (tileP + p)); pp++) {
        uint32_t m1 = m;
        for (;m1 < XTile.m(); m1++) {
          const ElemT* ptr = (isFirstFactor) ? 
                              XTile.data(m1, k/F.p(), tileP + p + pp) : 
                              &Ych.at(0, 0, 0) + (k + p + pp)*Ych.m() + m;
          ElemT val = *ptr;

          if (isLastFactor &&
              (EpilogueKindVal & EpilogueKind::Alpha) == EpilogueKind::Alpha) {
            val = alpha * val;
          }

          Xch.at(m1, k/F.p(), p + pp) = val;
        }}
      }
    }}
    
    if (!(kPMultipleOfTileP && Xch.p() % VecTLen == 0)) {
      uint32_t p = ((F.p() - tileP)/VecTLen) * VecTLen;
      for (; p < Xch.p(); p++) {
        uint32_t m = 0;
        if (p < F.p() - tileP) {
          for (;m < XTile.m(); m++) {
            const ElemT* ptr = (isFirstFactor) ? 
                                XTile.data(m, k/F.p(), tileP + p) : 
                                &Ych.at(0, 0, 0) + (k + p)*Ych.m() + m;
            
            ElemT val = *ptr;

            if (isLastFactor &&
                (EpilogueKindVal & EpilogueKind::Alpha) == EpilogueKind::Alpha) {
              val = alpha * val;
            }

            Xch.at(m, k/F.p(), p) = val;
          }
        }

        Xch.zero(m, k/F.p(), p,
                Xch.m(), k/F.p()+1, p+1);
    }
  }}
  }
}