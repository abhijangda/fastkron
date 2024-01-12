template<typename XReg, typename FReg, typename YReg>
CUDA_DEVICE
void slicedMMA(uint32_t rowA, XReg& Xr, FReg& Fr, YReg& Yr) {
  //Matrix Multiply Accumulate  
  #pragma unroll
  for (uint i = 0;    i < Yr.SliceM();         i++)
  #pragma unroll
  for (uint j = 0;    j < Yr.SliceN();         j++) {
    #pragma unroll
    for (uint k = 0;    k < Xr.TileP(); k++) {
      Yr.add(rowA, i, j, Xr.at(rowA, i, k) * Fr.at(k, j));
    }
  }
}

template<typename ElemT, typename XShared, typename FShared, 
         typename YReg, typename XReg, typename FReg>
CUDA_DEVICE
void mainMMA(XShared& Xsh, FShared& Fsh, YReg& Yr, XReg& Xr, FReg& Fr) {
  uint round_start = (Yr.yK / Yr.SliceM())%Xr.TileP();

  #pragma unroll
  for (uint rowA = 0; rowA < Yr.TileM(); rowA++) {
  if (rowA < Xsh.m()) {
    #pragma unroll
    for (uint rowC = 0; rowC < Xr.RegK(); rowC++) {
      uint shACol = Yr.yK + rowC;
      #pragma unroll
      for (uint colC = 0; colC < Xr.TileP(); colC++) {
        //TODO: bring shift calculation in Xsh.at
        ElemT temp = Xsh.template at<ElemT>(rowA, shACol * Xr.TileP() + (colC + round_start)%Xr.TileP());
        Xr.set(rowA, rowC, colC, temp);
      }
  }}}
  
  #pragma unroll
  for (uint colC = 0; colC < Yr.SliceN(); colC++) {
    uint shKronCol = Yr.yQ + colC;
    #pragma unroll
    for (uint elem = 0; elem < Xr.TileP(); elem++) {
      Fr.set(elem, colC, Fsh.template at<ElemT>(elem, shKronCol));
    }
  }

  #pragma unroll
  for (uint rowA = 0; rowA < Yr.TileM(); rowA++)
    if (rowA < Xsh.m())
      slicedMMA(rowA, Xr, Fr, Yr);
}