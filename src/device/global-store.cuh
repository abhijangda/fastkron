template<typename ElemT, typename DistParams>
CUDA_DEVICE
ElemT* p2pStoreAddress(const DistParams& distParams, const Matrix& Y,
                         uint32_t row, uint32_t col) {
  //TODO: Function do not need row
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

template<typename ElemT, typename FusedParams, typename ShiftShared>
CUDA_DEVICE
uint32_t fusedYColumn(const FusedParams& fusedParams, const Matrix& Y,
                      const ShiftShared& Xsh, const uint32_t tileK, const uint32_t Q, uint32_t xshCol) {
  const uint TileSizeColsAByKronCols = Xsh.n()/Q;
  uint withinP5 = tileK * fusedParams.UVAColsRatioKronColsSquare +
                  ((xshCol%TileSizeColsAByKronCols)/fusedParams.UVAColsRatioKronColsSquare)*fusedParams.ColsCByKronColsPower + 
                  xshCol%fusedParams.UVAColsRatioKronColsSquare;
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
template<typename ElemT>
CUDA_DEVICE
void stGlobalVec(ElemT* addr, int numValues, ElemT values[]) {
}

template<>
CUDA_DEVICE
void stGlobalVec(float* addr, int numValues, float values[]) {
  switch (numValues) {
    case 1:
      asm volatile ("st.global.f32 [%0], {%1};" ::
                    "l"(addr), "f"(values[0]));
      break;
    case 2:
      asm volatile ("st.global.v2.f32 [%0], {%1, %2};" ::
                    "l"(addr), "f"(values[0]), "f"(values[1]));
      break;
    case 4:
      asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
                    "l"(addr), "f"(values[0]), "f"(values[1]), "f"(values[2]), "f"(values[3]));
      break;
  }
}

template<uint32_t FusedMuls, uint32_t XAlign, uint32_t CRegRows>
CUDA_DEVICE
constexpr uint32_t storeVectorElems() {
  constexpr uint vecTyNumElems = (FusedMuls == 1) ? MIN(XAlign, MIN(CRegRows, 4) & (8 - 1)) : 1;
  static_assert (vecTyNumElems == 4 || vecTyNumElems == 2 || vecTyNumElems == 1);
  return vecTyNumElems;
}