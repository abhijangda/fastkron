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
uint32_t fusedYColumn(const FusedParams& params, const uint32_t fac, const Matrix& Y, const XShared& Xsh,
                      const uint32_t tileK, const uint32_t P, const uint32_t Q, const uint32_t xshCol) {
  const uint32_t XTileSlices = Xsh.n()/P;
  //Scale shared mem slice idx to global mem idx
  uint32_t glSlice = (xshCol/XTileSlices)*(Y.n()/Q);
  //Scale shared fused slice to global mem
  uint32_t sliceElem = ((xshCol%XTileSlices)/params.XShFusedSlices[fac])*params.XglFusedSlices[fac];
  //Elem idx in Fused Slice
  uint32_t elem = tileK * params.XShFusedSlices[fac] + xshCol%params.XShFusedSlices[fac];
  return glSlice + sliceElem + elem;
}

template<typename ElemT, typename EpilogueParams, typename GetBatchedData>
CUDA_DEVICE
ElemT epilogue(const EpilogueParams& params, GetBatchedData& batchedData, const Matrix& Y, uint32_t batch, uint32_t idx, ElemT yVal) {
  //Always reading struct members within the && condition is better than reading all before the condition.
  ElemT d = (params.template getBeta<ElemT>() != 0 &&
             batchedData.getZBatch(params, Y, batch).data() != nullptr) ? 
             params.template getBeta<ElemT>()*
             batchedData.getZBatch(params, Y, batch).template data<ElemT>(0)[idx] :
             0;
  return params.template getAlpha<ElemT>() * yVal + d;
}

template<fastKronOp OpY, uint32_t StLen, typename T, typename YReg>
CUDA_DEVICE
void getYVector(T* yreg, YReg& Yr, int row, int i, int j) {
  for (int e = 0; e < StLen; e++) {
    if (OpY == fastKronOp_N) {
      yreg[e] = Yr.at(row, i+e, j);
    } else if (OpY == fastKronOp_T) {
      yreg[e] = Yr.at(row+e, i, j);
    }
  }
}

//Store PTX instructions for each vector type
template<fastKronOp OpY, uint32_t StLen, typename YReg>
CUDA_DEVICE
void stVecYReg(float* addr, YReg& Yr, int row, int i, int j) {
  float yreg[StLen];
  getYVector<OpY, StLen, float>(yreg, Yr, row, i, j);

  switch (StLen) {
    case 1:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.f32 [%0], {%1};" ::
                    "l"(addr), 
                    "f"(yreg[0]));
    #elif defined(__HIPCC__)
      *addr = yreg[0];
    #endif
      break;
    case 2:
    {
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.v2.f32 [%0], {%1, %2};" ::
                    "l"(addr),
                    "f"(yreg[0]), "f"(yreg[1]));
    #elif defined(__HIPCC__)
      *addr = yreg[0];
      *(addr + 1) = yreg[1];
    #endif
      break;
    }
    case 4:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
                    "l"(addr), 
                    "f"(yreg[0]), "f"(yreg[1]), 
                    "f"(yreg[2]), "f"(yreg[3]));
    #elif defined(__HIPCC__)
      *addr = yreg[0];
      *(addr + 1) = yreg[1];
      *(addr + 2) = yreg[2];
      *(addr + 3) = yreg[3];
    #endif
      break;
  }
}

template<fastKronOp OpY, uint32_t StLen, typename YReg>
CUDA_DEVICE
void stVecYReg(int* addr, YReg& Yr, int row, int i, int j) {
  int yreg[StLen];
  getYVector<OpY, StLen, int>(yreg, Yr, row, i, j);

  switch (StLen) {
    case 1:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.s32 [%0], {%1};" ::
                    "l"(addr), 
                    "r"(yreg[0]));
    #elif defined(__HIPCC__)
      *addr = yreg[0];
    #endif
      break;
    case 2:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.v2.s32 [%0], {%1, %2};" ::
                    "l"(addr),
                    "r"(yreg[0]), "r"(yreg[1]));
    #elif defined(__HIPCC__)
      *addr = yreg[0];
      *(addr + 1) = yreg[1];
    #endif
      break;
    case 4:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.v4.s32 [%0], {%1, %2, %3, %4};" ::
                    "l"(addr),
                    "r"(yreg[0]), "r"(yreg[1]), 
                    "r"(yreg[2]), "r"(yreg[3]));
    #elif defined(__HIPCC__)
      *addr = yreg[0];
      *(addr + 1) = yreg[1];
      *(addr + 2) = yreg[2];
      *(addr + 3) = yreg[3];
    #endif
      break;
  }
}

template<fastKronOp OpY, uint32_t StLen, typename YReg>
CUDA_DEVICE
void stVecYReg(double* addr, YReg& Yr, int row, int i, int j) {
  double yreg[StLen];
  getYVector<OpY, StLen, double>(yreg, Yr, row, i, j);

  switch (StLen) {
    case 1:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.f64 [%0], {%1};" ::
                    "l"(addr),
                    "d"(yreg[0]));
    #elif defined(__HIPCC__)
      *addr = yreg[0];
    #endif
      break;
    case 2:
    #if defined(__NVCC__) || defined(__CUDACC__)
      asm volatile ("st.global.v2.f64 [%0], {%1, %2};" ::
                    "l"(addr),
                    "d"(yreg[0]), "d"(yreg[1]));
    #elif defined(__HIPCC__)
      *addr = yreg[0];
      *(addr + 1) = yreg[1];
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

template<fastKronOp OpY, uint32_t kMMultipleOfTileM, uint32_t kKMultipleOfTileK, 
         uint32_t FusedMuls, uint32_t XAlign, uint32_t RegM, uint32_t RegK>
CUDA_DEVICE
constexpr uint32_t storeVectorLen() {
  if (OpY == fastKronOp_N) {
    constexpr uint len = (kKMultipleOfTileK and FusedMuls == 1) ? 
                        MIN(XAlign, MIN(RegK, 4) & (8 - 1)) :
                        1;
    static_assert (len == 4 || len == 2 || len == 1);
    return len;
  } else if (OpY == fastKronOp_T) {
    constexpr uint len = (kMMultipleOfTileM and FusedMuls == 1) ?
                         MIN(XAlign, MIN(RegM, 4) & (8 - 1)) : 1;
    static_assert (len == 4 || len == 2 || len == 1);
    return len;
  }

  return 1;
}

template<fastKronOp OpY, uint32_t StLen, uint32_t TileQ,
         bool kMMultipleOfTileM, bool kKMultipleOfTileK, bool kQMultipleOfTileQ,
         bool isFused, bool DistributeToGPUs,
         typename ElemT, typename YElem, typename YReg, typename XTileTy, typename XShared,
         typename FusedParams, typename DistributedParams, typename EpilogueParams, typename GetBatchedData>
CUDA_DEVICE
void storeVectorY(const bool isLastFactor, const uint32_t fac, const uint32_t batch,
            const uint32_t rm, const uint32_t rq, const uint32_t rk,
            const uint32_t XshSlices, const uint32_t XSlices, 
            const uint32_t tileM, const uint32_t tileK, const uint32_t tileQ,
            const uint32_t P, const uint32_t Q,
            const XTileTy& XTile, const XShared& Xsh,
            const Matrix& Y, const YElem& yElem, YReg& yReg,
            const FusedParams& fusedParams, const DistributedParams& distParams,
            const EpilogueParams& epilogueParams, const GetBatchedData& batchedData) {
  const uint glM = rm + yElem.m() + tileM;
  if (!(kMMultipleOfTileM || (rm + yElem.m() < XTile.m()))) return;
  if ((!kKMultipleOfTileK && yElem.k() + rk >= MIN(XshSlices, XSlices - tileK * XshSlices)) || 
      (!kQMultipleOfTileQ && yElem.q() + rq >= MIN(TileQ, Q - tileQ * TileQ))) return;

  uint glK;
  ElemT* yPtr;
  uint32_t cIdx;

  //Total elements produced from TileK are (TileK/P) * Q
  //No. of elems produced by slice-multiply of TileK with
  //the same col of F are: TileK/P, i.e, XshSlices.
  //These elems are stored consecutively.
  if (isFused) {
    //Compute element location inside the tile
    const uint32_t shK = (yElem.q()   + rq) * // F's col multiplied by this thread
                          XshSlices +       // Index of first element produced by this F's col
                          yElem.k()   + rk ;  // index of element produced by multiplying this col with this slice
    glK = fusedYColumn(fusedParams, fac, Y, Xsh, tileK, P, Q, shK);
  } else {
    //Scale element location from within tile to global
    glK = (yElem.q()   + rq)  * //The index of elems by one column in TileK
            XSlices            + //Scale the index to global column
            tileK * XshSlices  + //Index of XshSlices elems produced by a tileK 
            yElem.k()    + rk;   //The element index within consecutive elems
    if (TileQ < Q) {
      glK += tileQ * XSlices * TileQ;
  }}


  if (DistributeToGPUs) {
    yPtr = p2pStoreAddress<ElemT, DistributedParams>(distParams, Y, glM, glK);
  } else {
    if (OpY == fastKronOp_N)
      cIdx = glM * Y.n() + glK;
    else
      cIdx = glK * Y.m() + glM;
    yPtr = Y.data<ElemT>(glM, glK, OpY);
    if (isLastFactor){
      #pragma unroll
      for (int i = 0; i < StLen; i++) {
        ElemT yelem = 0;
        if (OpY == fastKronOp_N) {
          yelem = yReg.at(rm, rk+i, rq);
        } else if (OpY == fastKronOp_T) {
          yelem = yReg.at(rm+i, rk, rq);
        }
        yelem = epilogue(epilogueParams, batchedData, Y, batch, cIdx+i, yelem);
        if (OpY == fastKronOp_N) {
          yReg.set(rm, rk + i, rq, yelem);
        } else if (OpY == fastKronOp_T) {
          yReg.set(rm+i, rk, rq, yelem);
        }
      }
    }
  }
  stVecYReg<OpY, StLen>(yPtr, yReg, rm, rk, rq);
}


template<fastKronOp OpY, uint32_t RegM, uint32_t RegK, uint32_t RegQ, 
         uint32_t TileQ,
         bool kMMultipleOfTileM, bool kKMultipleOfTileK, bool kQMultipleOfTileQ,
         uint32_t FusedFacs, bool DistributeToGPUs, uint32_t XAlignment,
         typename ElemT, typename YElem, typename YReg, typename XTileTy, typename XShared,
         typename KernelParams, typename FusedParams, typename DistributedParams, typename EpilogueParams, typename GetBatchedData>
CUDA_DEVICE
void storeY(const uint32_t fac, const uint32_t batch,
            const uint32_t XshSlices, const uint32_t XSlices, 
            const uint32_t tileM, const uint32_t tileK, const uint32_t tileQ,
            const uint32_t P, const uint32_t Q,
            const XTileTy& XTile, const XShared& Xsh,
            const Matrix& Y, const YElem& yElem, YReg& yReg,
            const KernelParams& params,
            const FusedParams& fusedParams, const DistributedParams& distParams,
            const EpilogueParams& epilogueParams, const GetBatchedData& batchedData) {
  constexpr uint32_t StLen = storeVectorLen<OpY, kMMultipleOfTileM, kKMultipleOfTileK, 
                                              FusedFacs, XAlignment, RegM, RegK>();

  if (OpY == fastKronOp_N) {
    #pragma unroll
    for (uint rm = 0; rm < RegM; rm++) {
    #pragma unroll
    for (uint tq = 0; tq < RegQ; tq++) {
    #pragma unroll
    for (uint tk = 0; tk < RegK; tk += StLen) {
      // if (params.kp_idx == 1) {assert(yReg.data[0] == 128); assert(yReg.data[1] == 128);}
      storeVectorY<OpY, StLen, TileQ,
             kMMultipleOfTileM, kKMultipleOfTileK, kQMultipleOfTileQ,
             (FusedFacs>1), DistributeToGPUs,
             ElemT>
        (params.kp_idx == ((FusedFacs == 1 ? 1 : params.problem.n()) - 1) && fac == 0,
         fac, batch,
         rm, tq, tk, XshSlices, XSlices,
         tileM, tileK, tileQ, P, Q,
         XTile, Xsh, Y, yElem, yReg,
         fusedParams, distParams, epilogueParams, batchedData);
    }}}
  } else if (OpY == fastKronOp_T) {
    #pragma unroll
    for (uint tq = 0; tq < RegQ; tq++) {
    #pragma unroll
    for (uint tk = 0; tk < RegK; tk++) {
    #pragma unroll
    for (uint rm = 0; rm < RegM; rm+=StLen) {
      storeVectorY<OpY, StLen, TileQ, 
             kMMultipleOfTileM, kKMultipleOfTileK, kQMultipleOfTileQ,
             (FusedFacs>1), DistributeToGPUs,
             ElemT>
        (epilogueParams.isLastFactor && fac == 0, fac, batch,
         rm, tq, tk, XshSlices, XSlices,
         tileM, tileK, tileQ, P, Q,
         XTile, Xsh, Y, yElem, yReg,
         fusedParams, distParams, epilogueParams, batchedData);
    }}}
  }
}