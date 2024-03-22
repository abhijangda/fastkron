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
  }
}

template<typename XShared, typename FShared, 
         typename YReg, typename XReg, typename FReg>
CUDA_DEVICE
void mainMMA(uint32_t stage, uint32_t m, XShared& Xsh, FShared& Fsh, YReg& Yr, XReg& Xr, FReg& Fr, const YElem& yElem) {
  //Load shared memory Xsh to registers Xr
  if (false) {
    #pragma unroll
    for (uint rm = 0; rm < Yr.m(); rm++) {
    if (rm < m) {
      #pragma unroll
      for (uint rk = 0; rk < Xr.k(); rk++) {
        uint shXk = yElem.k() + rk;
        uint shift = (yElem.k() / Yr.k())%Xr.p();

        #pragma unroll
        for (uint p = 0; p < Xr.p(); p++) {
          //TODO: bring shift calculation in Xsh.at
          auto temp = Xsh.at(stage, rm, shXk * Xr.p() + (p + shift)%Xr.p());
          Xr.set(rm, rk, p, temp);
        }
    }}}

    #pragma unroll
    for (uint rq = 0; rq < Yr.q(); rq++) {
      uint shFcol = yElem.q() + rq;
      #pragma unroll
      for (uint p = 0; p < Xr.p(); p++) {
        Fr.set(p, rq, Fsh.at(stage, p, shFcol));
      }
    }

    #pragma unroll
    for (uint rm = 0; rm < Yr.m(); rm++)
      slicedMMA(rm, Xr, Fr, Yr);
  }

  if (true) {
    const uint32_t RegP = 4;
    const uint32_t RegK = 4;
    const uint32_t RegQ = 1;

    float regX[1][1];
    float regF[1][1];
    using float4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
    float4 regY = {0,0,0,0};

    const uint32_t lane = threadIdx.x % 64;

    for (int rp = 0; rp < Xr.p(); rp += RegP) {
      for (uint rm = 0; rm < Yr.m(); rm++) {
        if (rm < m) {
          uint slice = lane / RegP;
          uint elem = lane % RegP;

          uint shift = (slice / RegK) % Xr.p();

          regX[0][0] = Xsh.at(stage, rm, slice * Xr.p() + (rp + elem + shift)%Xr.p());
      }}

      {
        uint row = lane / RegP;
        uint col = lane % RegP;

        regF[0][0] = Fsh.at(stage, col + rp, row);  
      }

      {
        regY = __builtin_amdgcn_mfma_f32_16x16x4f32(regX[0][0], regF[0][0], regY, 0,0,0);
        Yr.add(m, 0, 0, regY[0]);
        Yr.add(m, 1, 0, regY[1]);
        Yr.add(m, 2, 0, regY[2]);
        Yr.add(m, 3, 0, regY[3]);
      }
    }
  }
}