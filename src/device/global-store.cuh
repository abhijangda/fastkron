template<typename ElemT, typename DistParams>
__device__ __forceinline__
void p2pStoreAddress(const DistParams& distParams, const Matrix& Y,
                     uint32_t row, uint32_t col, ElemT*& ptr, uint32_t& addr) {
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
  addr = row * perGPUN + gpuCol;
  ptr = (ElemT*)(distParams.getLocalGPUResult(nextGc));
}

template<typename ElemT, typename FusedParams, typename ShiftShared>
__device__ __forceinline__
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
__device__ __forceinline__
ElemT epilogue(const EpilogueParams& params, uint32_t idx, ElemT yVal) {
  ElemT d = params.getBeta<ElemT>() * ((params.getD<ElemT>() != nullptr) ? params.getD<ElemT>()[idx] : 0);
  return params.getAlpha<ElemT>() * yVal + d;
}


//Store PTX instructions for each vector type
template<typename ElemT>
CUDA_DEVICE void stGlobalVec(ElemT* addr, int numValues, ElemT values[]) {
}

template<>
CUDA_DEVICE void stGlobalVec(float* addr, int numValues, float values[]) {
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