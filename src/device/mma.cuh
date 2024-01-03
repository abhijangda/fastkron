template<typename XReg, typename FReg, typename YReg>
__device__ __forceinline__ 
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

template<typename ElemT, typename XShared, typename FShared, typename YReg,
         uint32_t TileM, uint32_t CRegRows, uint32_t CRegCols, uint32_t TileP>
__device__ __forceinline__
void mainMMA(uint tileColA, uint32_t outerTileKronCol, XShared& Xsh, FShared& Fsh, YReg& Yr) {
  register XRegisters<ElemT, TileM, CRegRows, TileP> Xr;
  register FRegisters<ElemT, TileP, CRegCols> Fr;
  uint round_start = (tileColA / CRegRows)%TileP;

  #pragma unroll
  for (uint rowA = 0; rowA < Yr.TileM(); rowA++) {
  if (rowA < Xsh.m()) {
    #pragma unroll
    for (uint rowC = 0; rowC < CRegRows; rowC++) {
      uint shACol = tileColA + rowC;
      #pragma unroll
      for (uint colC = 0; colC < TileP; colC++) {
        ElemT temp = Xsh.template at<ElemT>(rowA, shACol * TileP + (colC + round_start)%TileP);
        Xr.set(rowA, rowC, colC, temp);
      }
  }}}
  
  #pragma unroll
  for (uint colC = 0; colC < CRegCols; colC++) {
    uint shKronCol = outerTileKronCol + colC;
    #pragma unroll
    for (uint elem = 0; elem < TileP; elem++) {
      Fr.set(elem, colC, Fsh.template at<ElemT>(elem, shKronCol));
    }
  }

  #pragma unroll
  for (uint rowA = 0; rowA < Yr.TileM(); rowA++)
    if (rowA < Xsh.m())
      slicedMMA(rowA, Xr, Fr, Yr);
}