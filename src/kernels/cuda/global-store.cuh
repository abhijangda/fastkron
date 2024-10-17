template<typename ElemT, typename DistParams>
CUDA_DEVICE
ElemT* p2pStoreAddress(const DistParams& distParams, const Matrix& Y,
                       uint32_t row, uint32_t col) {
  uint UVAColsRatioKronRowsSquare = distParams.UVAColsRatioKronRowsSquare;
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
uint32_t fusedYColumn(const FusedParams& params, const Matrix& Y, const XShared& Xsh,
                      const uint32_t tileK, const uint32_t P, const uint32_t Q, const uint32_t xshCol) {
  const uint32_t XTileSlices = Xsh.n()/P;
  //Scale shared mem slice idx to global mem idx
  uint32_t glSlice = (xshCol/XTileSlices)*(Y.n()/Q);
  //Scale shared fused slice to global mem
  uint32_t sliceElem = ((xshCol%XTileSlices)/params.XShFusedSlices)*params.XglFusedSlices;
  //Elem idx in Fused Slice
  uint32_t elem = tileK * params.XShFusedSlices + xshCol%params.XShFusedSlices;
  return glSlice + sliceElem + elem;
}

template<typename ElemT, typename EpilogueParams, typename GetBatchedData>
CUDA_DEVICE
ElemT epilogue(const EpilogueParams& params, GetBatchedData& batchedData, const Matrix& Y, uint32_t batch, uint32_t idx, ElemT yVal) {
  //Always reading struct members within the && condition is better than reading all before the condition.
  ElemT d = (params.template getBeta<ElemT>() != 0 && batchedData.getZBatch(params, Y, batch).data() != nullptr) ? 
             params.template getBeta<ElemT>() * batchedData.getZBatch(params, Y, batch).template data<ElemT>(0)[idx] :
             0;
  return params.template getAlpha<ElemT>() * yVal + d;
}


//Store PTX instructions for each vector type
template<typename YReg>
CUDA_DEVICE
void stVecYReg(float* addr, YReg& Yr, int numValues, int row, int i, int j) {
  switch (numValues) {
    case 1:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.f32 [%0], {%1};" ::
                    "l"(addr), 
                    "f"(Yr.at(row, i, j)));
    #elif defined(__HIPCC__)
      *addr = Yr.at(row, i, j);
    #endif
      break;
    case 2:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.v2.f32 [%0], {%1, %2};" ::
                    "l"(addr),
                    "f"(Yr.at(row, i+0, j)), "f"(Yr.at(row, i+1, j)));
    #elif defined(__HIPCC__)
      *addr = Yr.at(row, i+0, j);
      *(addr + 1) = Yr.at(row, i+1, j);
    #endif
      break;
    case 4:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
                    "l"(addr), 
                    "f"(Yr.at(row, i  , j)), "f"(Yr.at(row+1, i, j)), 
                    "f"(Yr.at(row+2, i, j)), "f"(Yr.at(row+3, i, j)));
    #elif defined(__HIPCC__)
      *addr = Yr.at(row, i+0, j);
      *(addr + 1) = Yr.at(row, i+1, j);
      *(addr + 2) = Yr.at(row, i+2, j);
      *(addr + 3) = Yr.at(row, i+3, j);
    #endif
      break;
  }
}

template<typename YReg>
CUDA_DEVICE
void stVecYReg(int* addr, YReg& Yr, int numValues, int row, int i, int j) {
  switch (numValues) {
    case 1:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.s32 [%0], {%1};" ::
                    "l"(addr), 
                    "r"(Yr.at(row, i, j)));
    #elif defined(__HIPCC__)
      *addr = Yr.at(row, i, j);
    #endif
      break;
    case 2:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.v2.s32 [%0], {%1, %2};" ::
                    "l"(addr),
                    "r"(Yr.at(row, i+0, j)), "r"(Yr.at(row, i+1, j)));
    #elif defined(__HIPCC__)
      *addr = Yr.at(row, i+0, j);
      *(addr + 1) = Yr.at(row, i+1, j);
    #endif
      break;
    case 4:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.v4.s32 [%0], {%1, %2, %3, %4};" ::
                    "l"(addr), 
                    "r"(Yr.at(row, i  , j)), "r"(Yr.at(row, i+1, j)), 
                    "r"(Yr.at(row, i+2, j)), "r"(Yr.at(row, i+3, j)));
    #elif defined(__HIPCC__)
      *addr = Yr.at(row, i+0, j);
      *(addr + 1) = Yr.at(row, i+1, j);
      *(addr + 2) = Yr.at(row, i+2, j);
      *(addr + 3) = Yr.at(row, i+3, j);
    #endif
      break;
  }
}

template<typename YReg>
CUDA_DEVICE
void stVecYReg(double* addr, YReg& Yr, int numValues, int row, int i, int j) {
  switch (numValues) {
    case 1:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.f64 [%0], {%1};" ::
                    "l"(addr), 
                    "d"(Yr.at(row, i, j)));
    #elif defined(__HIPCC__)
      *addr = Yr.at(row, i, j);
    #endif
      break;
    case 2:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.v2.f64 [%0], {%1, %2};" ::
                    "l"(addr),
                    "d"(Yr.at(row, i+0, j)), "d"(Yr.at(row, i+1, j)));
    #elif defined(__HIPCC__)
      *addr = Yr.at(row, i+0, j);
      *(addr + 1) = Yr.at(row, i+1, j);
    #endif
      break;
    case 4:
    // #if defined(__NVCC__) || defined(__CUDACC__)
    //   asm volatile ("st.global.v4.f64 [%0], {%1, %2, %3, %4};" ::
    //                 "l"(addr), 
    //                 "d"(Yr.at(row, i  , j)), "d"(Yr.at(row, i+1, j)), 
    //                 "d"(Yr.at(row, i+2, j)), "d"(Yr.at(row, i+3, j)));
    // #elif defined(__HIPCC__)
    //   *addr = Yr.at(row, i+0, j);
    //   *(addr + 1) = Yr.at(row, i+1, j);
    //   *(addr + 2) = Yr.at(row, i+2, j);
    //   *(addr + 3) = Yr.at(row, i+3, j);
    // #endif
      break;
  }
}

template<uint32_t kKMultipleOfTileK, uint32_t FusedMuls, uint32_t XAlign, uint32_t RegK>
CUDA_DEVICE
constexpr uint32_t storeVectorLen() {
  constexpr uint len = (kKMultipleOfTileK and FusedMuls == 1) ? 
                      MIN(XAlign, MIN(RegK, 4) & (8 - 1)) :
                      1;
  static_assert (len == 4 || len == 2 || len == 1);
  return len;
}