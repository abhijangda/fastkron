#pragma once

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
  if (fac > 0 || (Fch.p() <= F.p() && tileP < F.p() - Fch.p())) {
    YReg.apply([&](X86VecT& e, const uint32_t rm, const uint32_t rk, const uint32_t rq) {
      e.store(&Ych.at(y.m() + rm * YReg.mvec(), y.q() + rq, y.k()/Fch.p() + rk * YReg.kvec()));
    });
  } else {
    YReg.apply([&](X86VecT& e, const uint32_t rm, const uint32_t rk, const uint32_t rq) {
      constexpr bool kQMultipleOfTileQ = KernelOptimizations::IsQMultipleOfTileQ(OptLevel);
      constexpr bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);
      constexpr bool kMMultipleOfTileM = KernelOptimizations::IsMMultipleOfTileM(OptLevel);

      uint32_t slice = y.k()/Fch.p() + rk * YReg.kvec();

      if (!kKMultipleOfTileK && slice >= XTile.cols/F.p()) return;
      if (!kQMultipleOfTileQ && tileQ + y.q() + rq >= F.q()) return;

      const uint32_t XTileSlices = XTile.tileCols()/F.p();
      const uint32_t XSlices     = Y.n()/F.q();
      uint32_t yN;

      if (fusedParams.NumFused > 1) {
        uint32_t xshCol = (rq + y.q()) * XTileSlices + rk*YReg.kvec() + y.k()/Fch.p();
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

      if (kMMultipleOfTileM || y.m() + rm*YReg.mvec() < XTile.m()) {
        uint32_t numElems;
        if (YReg.layout() == fastKronOp_N) {
          uint32_t slices = (kKMultipleOfTileK &&
                            XTile.tileCols() % YReg.kvec() == 0) ? 
                            YReg.kvec() : (XTile.cols/F.p() - slice);
          slices = MIN(YReg.kvec(), slices);
          numElems = slices;
        } else {
          numElems = kMMultipleOfTileM ? YReg.mvec() : XTile.m() - (y.m() + rm*YReg.mvec());
          numElems = MIN(YReg.mvec(), numElems);
        }
        if ((EpilogueKindVal & EpilogueKind::Beta) == EpilogueKind::Beta) {
          X86VecT z;
          z.load(Z.data<ElemT>(tileM + y.m() + rm*YReg.mvec(), yN, YReg.layout()), numElems);
          e.fmadd(beta, z);
        }
        e.store(Y.data<ElemT>(tileM + y.m() + rm*YReg.mvec(), yN, YReg.layout()), numElems);
    }});
  }
}