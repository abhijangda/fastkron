template<typename ElemT, typename DistParams>
CUDA_DEVICE
ElemT* p2pStoreAddress(const DistParams& distParams, const Matrix& Y,
                       uint32_t row, uint32_t col) {
  uint UVAColsRatioKronRowsSquare = distParams.UVAColsRatioKronRowsSquare;//(perGPUK/KronRowsPower); //
  const uint perGPUNByNumGPUs = distParams.perGPUNByNumGPUs;
  const uint perGPUNByKronCols = distParams.perGPUNByKronCols;
  const uint ColsCByKronCols = distParams.ColsCByKronCols;
  const uint gcMulUVAColsRatioKronRowsSquare = distParams.gcMulUVAColsRatioKronRowsSquare;
  const uint ColsCByKronColsPower = distParams.ColsCByKronColsPower;
  
  uint nextGc = col/perGPUNByNumGPUs;

  const uint perGPUN = Y.n();
  uint srcElem = col;
  uint withinP5 = gcMulUVAColsRatioKronRowsSquare +
                  ((srcElem%perGPUNByKronCols)/UVAColsRatioKronRowsSquare)*ColsCByKronColsPower +
                  srcElem % UVAColsRatioKronRowsSquare;
  uint p5Index = (srcElem/perGPUNByKronCols)*ColsCByKronCols;
  int newcCol = p5Index + withinP5;
  int gpuCol = newcCol - nextGc * perGPUN;
  uint32_t addr = row * perGPUN + gpuCol;
  ElemT* ptr = (ElemT*)(distParams.getLocalGPUResult(nextGc));
  return &ptr[addr];
}

template<typename FusedParams, typename XShared>
CUDA_DEVICE
uint32_t fusedYColumn(const FusedParams& fusedParams, const Matrix& Y, const XShared& Xsh,
                      const uint32_t tileK, const uint32_t Q, const uint32_t xshCol) {
  const uint TileSizeColsAByKronCols = Xsh.n()/Q;
  uint withinP5 = tileK * fusedParams.UVAColsRatioKronRowsSquare +
                  ((xshCol%TileSizeColsAByKronCols)/fusedParams.UVAColsRatioKronRowsSquare)*fusedParams.ColsCByKronColsPower + 
                  xshCol%fusedParams.UVAColsRatioKronRowsSquare;
  uint p5Index = (xshCol/TileSizeColsAByKronCols)*(Y.n()/Q);
  uint cCol = p5Index + withinP5;
  return cCol;
}

template<typename ElemT>
CUDA_DEVICE
ElemT epilogue(const EpilogueParams& params, uint32_t idx, ElemT yVal) {
  ElemT d = params.getBeta<ElemT>() * ((params.getD<ElemT>() != nullptr) ? params.getD<ElemT>()[idx] : 0);
  return params.getAlpha<ElemT>() * yVal + d;
}


//Store PTX instructions for each vector type
template<typename ElemT, typename YReg>
CUDA_DEVICE
void stVecYReg(ElemT* addr, YReg& Yr, int numValues, int row, int i, int j) {
  switch (numValues) {
    case 1:
      asm volatile ("st.global.f32 [%0], {%1};" ::
                    "l"(addr), 
                    "f"(Yr.at(row, i, j)));
      break;
    case 2:
      asm volatile ("st.global.v2.f32 [%0], {%1, %2};" ::
                    "l"(addr),
                    "f"(Yr.at(row, i+0, j)), "f"(Yr.at(row, i+1, j)));
      break;
    case 4:
      asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
                    "l"(addr), 
                    "f"(Yr.at(row, i  , j)), "f"(Yr.at(row, i+1, j)), 
                    "f"(Yr.at(row, i+2, j)), "f"(Yr.at(row, i+3, j)));
      break;
  }
}

template<uint32_t FusedMuls, uint32_t XAlign, uint32_t RegK>
CUDA_DEVICE
constexpr uint32_t storeVectorLen() {
  constexpr uint len = (FusedMuls == 1) ? 
                      MIN(XAlign, MIN(RegK, 4) & (8 - 1)) :
                      1;
  static_assert (len == 4 || len == 2 || len == 1);
  return len;
}