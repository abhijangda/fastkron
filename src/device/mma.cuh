template<typename XReg, typename FReg, typename YReg>
CUDA_DEVICE
void slicedMMA(uint32_t m, XReg& Xr, FReg& Fr, YReg& Yr) {
  //Matrix Multiply Accumulate  
  #pragma unroll
  for (uint i = 0; i < Yr.k(); i++)
  #pragma unroll
  for (uint j = 0; j < Yr.q(); j++)
  #pragma unroll
  for (uint p = 0; p < Xr.p();  p++) {
    Yr.add(m, i, j, Xr.at(m, i, p) * Fr.at(p, j));
  }
}

template<typename ElemT, typename XShared, typename FShared, 
         typename YReg, typename XReg, typename FReg>
CUDA_DEVICE
void mainMMA(XShared& Xsh, FShared& Fsh, YReg& Yr, XReg& Xr, FReg& Fr) {
  uint round_start = (Yr.yK / Yr.k())%Xr.p();

  //Load shared memory Xsh to registers Xr 
  #pragma unroll
  for (uint rm = 0; rm < Yr.m(); rm++) {
  if (rm < Xsh.m()) {
    #pragma unroll
    for (uint rk = 0; rk < Xr.k(); rk++) {
      uint shACol = Yr.yK + rk;
      #pragma unroll
      for (uint p = 0; p < Xr.p(); p++) {
        //TODO: bring shift calculation in Xsh.at
        ElemT temp = Xsh.template at<ElemT>(rm, shACol * Xr.p() + (p + round_start)%Xr.p());
        Xr.set(rm, rk, p, temp);
      }
  }}}
  
  #pragma unroll
  for (uint rq = 0; rq < Yr.q(); rq++) {
    uint shKronCol = Yr.yQ + rq;
    #pragma unroll
    for (uint p = 0; p < Xr.p(); p++) {
      Fr.set(p, rq, Fsh.template at<ElemT>(p, shKronCol));
    }
  }

  #pragma unroll
  for (uint rm = 0; rm < Yr.m(); rm++)
    if (rm < Xsh.m())
      slicedMMA(rm, Xr, Fr, Yr);
}