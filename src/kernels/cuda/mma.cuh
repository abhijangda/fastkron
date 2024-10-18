template<typename XReg, typename FReg, typename YReg>
CUDA_DEVICE
void slicedMMA(XReg& Xr, FReg& Fr, YReg& Yr) {
  //Matrix Multiply Accumulate
  #pragma unroll
  for (uint j = 0; j < Yr.q(); j++)
  #pragma unroll
  for (uint m = 0; m < Yr.m(); m++)
  #pragma unroll
  for (uint i = 0; i < Yr.k(); i++)
  #pragma unroll
  for (uint p = 0; p < Xr.p(); p++) {
    Yr.add(m, i, j, Xr.at(m, i, p) * Fr.at(p, j));
  }
}

template<typename XShared, typename FShared, 
         typename YReg, typename XReg, typename FReg>
CUDA_DEVICE
void mainMMA(uint32_t kp_idx, uint32_t m, XShared& Xsh, FShared& Fsh, YReg& Yr, XReg& Xr, FReg& Fr, const YElem& yElem) {
  //Load shared memory Xsh to registers Xr 
  if (Xsh.layout() == fastKronOp_N) {
    #pragma unroll
    for (uint rm = 0; rm < Yr.m(); rm++) {
    // if (rm < m) {
      #pragma unroll
      for (uint rk = 0; rk < Xr.k(); rk++) {
        uint shXk = yElem.k() + rk;
        uint shift = (yElem.k() / Yr.k());

        #pragma unroll
        for (uint p = 0; p < Xr.p(); p++) {
          //TODO: bring shift calculation in Xsh.at
          //TODO: use the actual type not float
          auto temp = Xsh.at(yElem.m() + rm, shXk * Xr.p() + (p + shift)%Xr.p());
          Xr.set(rm, rk, p, temp);
        // }
    }}}
  } else {
    #pragma unroll
    for (uint rk = 0; rk < Xr.k(); rk++) {
      uint shXk = yElem.k() + rk;
      uint shift = 0;//(yElem.k() / Yr.k());

      #pragma unroll
      for (uint p = 0; p < Xr.p(); p++) {  
        #pragma unroll
        for (uint rm = 0; rm < Yr.m(); rm++) {
          //TODO: bring shift calculation in Xsh.at
          float temp = Xsh.at((yElem.m() + rm + shift)/*%Xsh.m()*/, shXk * Xr.p() + p);
          Xr.set(rm, rk, p, temp);
        // }
    }}}
  }
  
  if (Fsh.layout() == fastKronOp_N) {
    #pragma unroll
    for (uint rq = 0; rq < Yr.q(); rq++) {
      uint shFcol = yElem.q() + rq;
      #pragma unroll
      for (uint p = 0; p < Xr.p(); p++) {
        Fr.set(p, rq, Fsh.at(p, shFcol));
    }}
  } else if (Fsh.layout() == fastKronOp_T) {
    uint32_t qe = yElem.q();
    #pragma unroll
    for (uint rq = 0; rq < Yr.q(); rq++) {
      uint32_t shFcol = qe + rq;
      #pragma unroll
      for (uint p = 0; p < Xr.p(); p++) {
        uint32_t shift = p;
        if (false) //Shift
          Fr.set(p, rq, (&Fsh.at(0,0))[(shift+shFcol)%Fsh.q() + p*Fsh.q()]);
        if (true) {//Padding
          auto fv = (&Fsh.at(0,0))[shFcol + p*(Fsh.q() + 1)];
          Fr.set(p, rq, (&Fsh.at(0,0))[shFcol + p*(Fsh.q() + 1)]);
        }
    }}
  }

  slicedMMA(Xr, Fr, Yr);
}