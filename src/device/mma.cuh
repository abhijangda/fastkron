template<typename XReg, typename FReg, typename YReg>
__device__ __forceinline__ 
void mainMMA

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