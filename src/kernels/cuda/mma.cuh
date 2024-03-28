template<typename XReg, typename FReg, typename YReg>
CUDA_DEVICE
void slicedMMA(uint32_t m, XReg& Xr, FReg& Fr, YReg& Yr) {
  //Matrix Multiply Accumulate  
  #pragma unroll
  for (uint i = 0; i < Yr.k(); i++)
  #pragma unroll
  for (uint j = 0; j < Yr.q(); j++)
  #pragma unroll
  for (uint p = 0; p < Xr.p(); p++) {
    Yr.add(m, i, j, Xr.at(m, i, p) * Fr.at(p, j));
    // if (threadIdx.x == 15 && blockIdx.x == 1 && blockIdx.y == 0 && i == 2 && j == 0) {
    //   printf("%f %f %f\n", Yr.at(0, 2, 0), Xr.at(0, 2, p), Fr.at(p, 0));
    // }
  }
}

template<bool kExactShapes, typename XShared, typename FShared, 
         typename YReg, typename XReg, typename FReg>
CUDA_DEVICE
void mainMMA(uint32_t m, XShared& Xsh, FShared& Fsh, YReg& Yr, XReg& Xr, FReg& Fr, const YElem& yElem, bool canPrint) {
  //Load shared memory Xsh to registers Xr 
  #pragma unroll
  for (uint rm = 0; rm < Yr.m(); rm++) {
  if (rm < m) {
    #pragma unroll
    for (uint rk = 0; rk < Xr.k(); rk++) {
      uint shXk = yElem.k() + rk;
      uint shift = shXk / Yr.k();

      #pragma unroll
      for (uint p = 0; p < Xr.p(); p++) {
        //TODO: bring shift calculation in Xsh.at
        float temp = 0.0f;
        temp = Xsh.at(rm, shXk * Xr.p() + (p + shift)%Xr.p());
        Xr.set(rm, rk, p, temp);
      }
  }}}
  
  #pragma unroll
  for (uint rq = 0; rq < Yr.q(); rq++) {
    uint shFcol = yElem.q() + rq;
    #pragma unroll
    for (uint p = 0; p < Xr.p(); p++) {
      // if (kExactShapes || shFcol < Fsh.q()) //TODO: Need to add these conditions outside of mainMMA
        Fr.set(p, rq, Fsh.at(p, shFcol));
      // else
        // Fr.set(p, rq, 0);
  }}

  #pragma unroll
  for (uint rm = 0; rm < Yr.m(); rm++)
    slicedMMA(rm, Xr, Fr, Yr);
}