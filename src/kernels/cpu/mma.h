#pragma once

template<typename X86VecT, 
         typename FCache, typename YInterim,
         typename YRegisters>
static CUDA_DEVICE_HOST
void loadYInterim(uint32_t tileP, const YElem& y,
          const FCache& Fch, YInterim& Ych, YRegisters& YReg) {
  if (tileP == 0) {
    YReg.zero();
  } else {
    //TODO: For OpY=fastKronOp_T YReg.apply should have last loop in m
    YReg.apply([&](X86VecT& e, const uint32_t ym, const uint32_t yk, const uint32_t yq) {
      e.load(&Ych.at(y.m() + ym * YReg.mvec(), y.q() + yq, y.k()/Fch.p() + yk * YReg.kvec()));
    });
  }
}

template<typename X86VecT, 
         typename XCache, typename FCache, typename YInterim,
         typename YRegisters>
static CUDA_DEVICE_HOST
void mma(uint32_t /*tileP*/, const YElem& y, 
         const XCache& Xch, const FCache& Fch,
         YInterim& /*Ych*/, YRegisters& YReg) {
  const fastKronOp Layout = YRegisters::layout();

  for (uint32_t p = 0; p < Fch.p(); p++) {
    XRegisters<Layout, X86VecT, YRegisters::m(), YRegisters::k(), 1> XReg;
    FRegisters<X86VecT, 1, YRegisters::q()> FReg;
    XReg.apply([&](X86VecT& e, const uint32_t em, const uint32_t ek, const uint32_t ep) {
      e.load(&Xch.at(y.m() + em*YReg.mvec(), y.k()/Fch.p() + ek*YReg.kvec(), p + ep));
    });

    FReg.apply([&](X86VecT& e, const uint32_t ep, const uint32_t eq) {
      e.broadcast(&Fch.at(p + ep, y.q() + eq));
    });

    YReg.apply([&](X86VecT& e, const uint32_t ym, const uint32_t yk, const uint32_t yq) {
      e.fmadd(XReg.at(ym, yk, 0), FReg.at(0, yq));
    });
  }
}