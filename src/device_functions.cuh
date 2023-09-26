template<uint MaxKronCols, uint MaxTileSizeKronCols> __device__ uint get_tile_k() {return blockIdx.x/DIVUP(MaxKronCols, MaxTileSizeKronCols);}
template<uint MaxKronCols, uint MaxTileSizeKronCols> __device__ uint get_external_tile_kp_n() {return blockIdx.x%DIVUP(MaxKronCols, MaxTileSizeKronCols);}

__device__ bool isfirstIdx(dim3 idx) {return idx.x == 0 && idx.y == 0 & idx.z == 0;}

__device__ constexpr uint sqrt(uint x) {
  switch (x) {
    case 1:
      return 1;

    case 2:
      return 2;
    
    case 4:
      return 2;
    
    case 8:
      return 4;
    
    case 16:
      return 4;
    
    case 32:
      return 8;
    
    case 64:
      return 8;
    
    default:
      return 1;
  }
}

//Compute x**y
__device__ __host__ constexpr uint power(const uint x, const uint y) {
  uint result = 1;
  for (uint i = 0; i < y; i++) {
    result = result * x;
  }
  return result;
}

template<uint x, uint y>
__device__ __host__ constexpr uint iconstpower() {
  uint result = 1;
  for (uint i = 0; i < y; i++) {
    result = result * x;
  }
  return result;
}

template<typename VecT, typename ElemT>
__device__ void globalLoadVec(const ElemT* addr, VecT& vec) {
  //Not implemented
}

template<>
__device__ void globalLoadVec(const float* addr, float4& vec) {
  asm ("ld.ca.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(vec.x), "=f"(vec.y), "=f"(vec.z), "=f"(vec.w) : "l"(addr));
}

template<>
__device__ void globalLoadVec(const int* addr, int4& vec) {
  vec = *(int4*)addr;
}

template<>
__device__ void globalLoadVec(const double* addr, double4& vec) {
  vec = *(double4*)addr;
}

template<>
__device__ void globalLoadVec(const float* addr, float& vec) {
  vec = *addr;
}

template<typename VecT, typename ElemT>
__device__ void loadVecToRegs(VecT& vec, ElemT* regs) {
  //Not implemented
}

//Four Element Vectors
template<typename VecT, typename ElemT>
__device__ void load4ElemVecToRegs(VecT& vec, ElemT* regs) {
  regs[0] = vec.x;
  regs[1] = vec.y;
  regs[2] = vec.z;
  regs[3] = vec.w;
}

template<>
__device__ void loadVecToRegs(float4& vec, float* regs) {
  load4ElemVecToRegs(vec, regs);
}

template<>
__device__ void loadVecToRegs(int4& vec, int* regs) {
  load4ElemVecToRegs(vec, regs);
}


template<>
__device__ void loadVecToRegs(double4& vec, double* regs) {
  load4ElemVecToRegs(vec, regs);
}

//Two element vectors
template<>
__device__ void loadVecToRegs(double2& vec, double* regs) {
  regs[0] = vec.x;
  regs[1] = vec.y;
}


//Single element
template<>
__device__ void loadVecToRegs(float& vec, float* regs) {
  regs[0] = vec;
}

//Store PTX instructions for each vector type
template<typename ElemT>
__device__ void globalStore4Elems(ElemT* addr, ElemT elem1, ElemT elem2, ElemT elem3, ElemT elem4) {
}

template<>
__device__ void globalStore4Elems(float* addr, float elem1, float elem2, float elem3, float elem4) {
  // asm ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "=f"(elem1), "=f"(elem2), "=f"(elem3), "=f"(elem4));
  float4 vec = {elem1, elem2, elem3, elem4};
  *(float4*)addr = vec;
}

template<>
__device__ void globalStore4Elems(int* addr, int elem1, int elem2, int elem3, int elem4) {
  // asm ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "=f"(elem1), "=f"(elem2), "=f"(elem3), "=f"(elem4));
  int4 vec = {elem1, elem2, elem3, elem4};
  *(int4*)addr = vec;
}

template<>
__device__ void globalStore4Elems(double* addr, double elem1, double elem2, double elem3, double elem4) {
  // asm ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "=f"(elem1), "=f"(elem2), "=f"(elem3), "=f"(elem4));
  double4 vec = {elem1, elem2, elem3, elem4};
  *(double4*)addr = vec;
}

template<typename ElemT>
__device__ void globalStore2Elems(ElemT* addr, ElemT elem1, ElemT elem2) {
}

template<>
__device__ void globalStore2Elems(float* addr, float elem1, float elem2) {
  float2 vec = {elem1, elem2};
  *(float2*)addr = vec;
}

template<>
__device__ void globalStore2Elems(int* addr, int elem1, int elem2) {
  int2 vec = {elem1, elem2};
  *(int2*)addr = vec;
}

template<>
__device__ void globalStore2Elems(double* addr, double elem1, double elem2) {
  double2 vec = {elem1, elem2};
  *(double2*)addr = vec;
}

template<typename ElemT>
__device__ void globalStore1Elems(ElemT* addr, ElemT elem1) {
  *addr = elem1;
}

template<typename ElemT>
__global__ void printArrayKernel(const uint Rows, const uint Cols, const ElemT val, const ElemT* array) {
  const uint row = blockIdx.x;
  uint col = threadIdx.x;
  if (threadIdx.x == 0) {
    for (; col < Cols; col++) {
      const uint id = row * Cols + col;
      if (row == 0 and col <= Cols and array[id] != val)
        printf("array[%d*%d + %d] %f\n", row, Cols, col, array[id]);
    }
  }
}

template<typename ElemT>
void printGPUArray(const uint Rows, const uint Cols, const ElemT val, const ElemT* array, cudaStream_t stream) {
  dim3 grid = {Rows, 1, 1};
  dim3 block = {256, 1, 1};

  printArrayKernel<ElemT><<<grid, block, 0, stream>>>(Rows, Cols, val, array);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}


template<typename ElemT, typename VecT, uint NumThreads>
__global__ void copyXtoUVAX(const uint RowsC,    const uint ColsC,   const uint ColsA,
                            const uint KronRows, const uint KronCols,
                            ElemT * __restrict__ uvaTemp,
                            const uint uvaRows, const uint uvaCols,
                            const ElemT * __restrict__ glA,
                            const uint uvaPart) {
  const uint WarpSize     = 32;
  const uint tid          = threadIdx.x;
  const uint wid          = tid/WarpSize;
  const uint lane         = tid%WarpSize;
  const uint blockWarps   = blockDim.x/WarpSize;
  
  const uint rowA = blockIdx.x;

  for (uint uvaElem = tid; uvaElem < uvaCols; uvaElem += NumThreads) {
    uvaTemp[rowA * uvaCols + uvaElem] = glA[rowA* ColsA + uvaPart * uvaCols + uvaElem];  
  }
}

template<typename ElemT, typename VecT, uint NumThreads>
__global__ void copyUVATempToY(const uint RowsC,    const uint ColsC,   const uint ColsA,
                            const uint KronRows, const uint KronCols,
                            ElemT * __restrict__ uvaTemp,
                            const uint uvaRows, const uint uvaCols,
                            ElemT * __restrict__ glC,
                            const uint uvaPart, const uint batchedKronMuls, const uint startKronIdx) {
  const uint WarpSize     = 32;
  const uint tid          = threadIdx.x;
  const uint wid          = tid/WarpSize;
  const uint lane         = tid%WarpSize;
  const uint blockWarps   = blockDim.x/WarpSize;
  const uint rowA = blockIdx.x;

  for (uint uvaElem = tid; uvaElem < uvaCols; uvaElem += NumThreads) {
    // uint cCol = outerTileKronCol*(MaxColsA/MaxKronRows) + reg_j*(MaxColsA/MaxKronRows) + shVecI;
    // //(0,0,0,0,0,16,16,16)*128 + (0,1,2,3,..16)*128
    // if (!K_EQUALS_VAR) {
    //   uint tile_k = get_tile_k<MaxKronCols, MaxTileSizeKronCols>();
    //   cCol = tile_k * (MaxColsA/kronCols) + 
    //       (cCol/(MaxColsA/kronCols)) * (colsA/kronCols) +
    //       cCol%(MaxColsA/kronCols);
    // }
    
    if (batchedKronMuls == 1) {
      uint cCol = uvaPart * (uvaCols/KronRows) + (uvaElem/(uvaCols/KronRows))*(ColsC/KronRows) + uvaElem%(uvaCols/KronRows);
      glC[rowA * ColsA + cCol] = uvaTemp[rowA * uvaCols + uvaElem];
    } else {
      uint KronRowsPower = power(KronRows, batchedKronMuls);
      
      uint UVAColsRatioKronRowsSquare = (uvaCols/KronRowsPower);
      uint withinP5 = uvaPart * UVAColsRatioKronRowsSquare + 
                      ((uvaElem%(uvaCols/KronRows))/UVAColsRatioKronRowsSquare)*(ColsC/(uvaCols/UVAColsRatioKronRowsSquare)) + 
                      uvaElem % UVAColsRatioKronRowsSquare;
      uint p5Index = (uvaElem/(uvaCols/KronRows))*(ColsA/KronRows);
      uint cCol = p5Index + withinP5; //(uvaElem/(uvaCols/KronRows))*(ColsC/KronRows) + uvaElem%(uvaCols/KronRows);
      glC[rowA * ColsA + cCol] = uvaTemp[rowA * uvaCols + uvaElem];
    }
  }
}

template<typename ElemT, typename VecT, uint NumGPUs, uint PerGPUK, uint TileK, uint NumThreads>
__launch_bounds__(NumThreads)
__global__ void copyToGPUsInK(const uint RowsC,    const uint ColsC,   const uint ColsA,
                               ElemT* __restrict__ gpuPrevResult,
                               ElemT* __restrict__ gpuResult1, ElemT* __restrict__ gpuResult2,
                               uint gr, uint gc,
                               uint KronMulBatchSize) {
  const uint WarpSize     = 32;
  const uint tid          = threadIdx.x;
  const uint wid          = tid/WarpSize;
  const uint lane         = tid%WarpSize;
  const uint blockWarps   = blockDim.x/WarpSize;
  const uint rowA = blockIdx.y;
  const uint tileK = (gc == 0) ? blockIdx.x * TileK : ((gridDim.x-blockIdx.x)*TileK);
  const uint KronRows = 64;
  const uint KronCols = 64;
  const uint ldElems = sizeof(VecT)/sizeof(ElemT);
  const uint ElemBatch = ldElems;

  for (uint e = tid * ElemBatch; e < TileK; e += NumThreads * ElemBatch) {
    uint elem = tileK + e; 
    const uint nextGc = elem/((PerGPUK/NumGPUs));
    uint batchedKronMuls = (KronMulBatchSize == 3) ? 3 : 1;
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 0) printf("batchedKronMuls %d\n", batchedKronMuls);
    uint KronRowsPower = (batchedKronMuls == 3) ? KronRows*KronRows*KronRows: KronRows; //power(kronRows, batchedKronMuls);

    uint srcElem = elem;
    uint UVAColsRatioKronRowsSquare = (PerGPUK/KronRowsPower);
    uint withinP5 = gc * UVAColsRatioKronRowsSquare + 
                    ((srcElem%(PerGPUK/KronRows))/UVAColsRatioKronRowsSquare)*(ColsC/(PerGPUK/UVAColsRatioKronRowsSquare)) + 
                    srcElem % UVAColsRatioKronRowsSquare;
    uint p5Index = (srcElem/(PerGPUK/KronRows))*(ColsA/KronRows);
    int newcCol = p5Index + withinP5;
    int gpuCol = newcCol - nextGc * PerGPUK;
    auto outputArray = (nextGc == 0) ? gpuResult1 : gpuResult2;

    for (uint batchElem = 0; batchElem < ElemBatch; batchElem += ldElems) {
      VecT r = *((VecT*)&gpuPrevResult[rowA*PerGPUK + elem + batchElem]);
      *((VecT*)&outputArray[rowA * PerGPUK + gpuCol + batchElem]) = r;      
    }
  }
}

template<typename ElemT, typename VecT, uint NumThreads>
__global__ void storeGPUTile(const uint RowsC,    const uint ColsC,   const uint ColsA,
                             const uint KronRows, const uint KronCols,
                             const uint rank, const uint numGPUs,
                             ElemT * __restrict__ slicedGPUOutput,
                             const uint perGPUM, const uint perGPUK,
                             ElemT * __restrict__ gpuOutput,
                             const uint srcRank, const uint batchedKronMuls, const uint startKronIdx, bool canPrint) {
  const uint WarpSize     = 32;
  const uint tid          = threadIdx.x;
  const uint wid          = tid/WarpSize;
  const uint lane         = tid%WarpSize;
  const uint blockWarps   = blockDim.x/WarpSize;
  const uint rowA = blockIdx.x;

  for (uint elem = tid; elem < perGPUK/numGPUs; elem += NumThreads) {
    // uint cCol = outerTileKronCol*(MaxColsA/MaxKronRows) + reg_j*(MaxColsA/MaxKronRows) + shVecI;
    // //(0,0,0,0,0,16,16,16)*128 + (0,1,2,3,..16)*128
    // if (!K_EQUALS_VAR) {
    //   uint tile_k = get_tile_k<MaxKronCols, MaxTileSizeKronCols>();
    //   cCol = tile_k * (MaxColsA/kronCols) + 
    //       (cCol/(MaxColsA/kronCols)) * (colsA/kronCols) +
    //       cCol%(MaxColsA/kronCols);
    // }
    
    if (batchedKronMuls == 1) {
      uint srcElem = rank * (perGPUK/numGPUs) + elem;
      uint globalCCol = srcRank * (perGPUK/KronRows);
      globalCCol += (srcElem/(perGPUK/KronRows))*(ColsC/KronRows);
      globalCCol += srcElem%(perGPUK/KronRows);
      int gpuCol = globalCCol - rank * perGPUK;
      const uint id = rowA * (perGPUK/numGPUs) + elem;
      auto e = slicedGPUOutput[id];
      // if (canPrint and rowA == 1) printf("rowA %d gpuCol %d e %f id %d ColsA %d\n", rowA, gpuCol, e, id, ColsA);
      // if (rowA == 0 && rank == 0 && srcRank == 0) printf("gpuCol %d globalCCol %d\n", gpuCol, globalCCol);
      // if (rowA == 0)
      //   printf("rowA %d perGPUK %d numGPUs %d elem %d ColsA %d gpuCol %d id %d e %f\n", rowA, perGPUK, numGPUs, elem, ColsA, gpuCol, id, e);
      gpuOutput[rowA * perGPUK + gpuCol] = e;
    } else {
      uint KronRowsPower = power(KronRows, batchedKronMuls);
      uint srcElem = rank * (perGPUK/numGPUs) + elem;
      uint UVAColsRatioKronRowsSquare = (perGPUK/KronRowsPower);
      uint withinP5 = srcRank * UVAColsRatioKronRowsSquare + 
                      ((srcElem%(perGPUK/KronRows))/UVAColsRatioKronRowsSquare)*(ColsC/(perGPUK/UVAColsRatioKronRowsSquare)) + 
                      srcElem % UVAColsRatioKronRowsSquare;
      uint p5Index = (srcElem/(perGPUK/KronRows))*(ColsA/KronRows);
      uint cCol = p5Index + withinP5;
      int gpuCol = cCol - rank * perGPUK;

      gpuOutput[rowA * perGPUK + gpuCol] = slicedGPUOutput[rowA * (perGPUK/numGPUs) + elem];
    }
  }
}

template<typename ElemT, 
        const uint TileCol, const uint TileRow, const uint NumThreads>
__global__ void matrixSliceKernel(const uint Rows, const uint Cols, const ElemT* matrix, 
                            const uint startRow, const uint startCol, 
                            const uint SliceRows, const uint SliceCols,
                            ElemT* output) {
  // uint blockCol = blockIdx.x * TileCol;
  uint blockRow = blockIdx.x * TileRow;
  static_assert (TileRow == 1 and TileCol == 1);
  // if (startCol + SliceCols >= Cols || startRow + blockRow >= Rows) return;
  const uint matrixRow = startRow + blockRow;
  for (uint col = threadIdx.x; col < SliceCols; col += NumThreads) {
    const uint matrixCol = startCol + 0 + col;
    output[blockRow * SliceCols + col] = matrix[matrixRow*Cols + matrixCol];
    // if (threadIdx.x == 0)
    // printf("301: row %d SliceCols %d col %d id %d matrixRow %d Cols %d matrixCol %d output %f matrix %f\n", row, SliceCols, col, row * SliceCols + col, matrixRow, Cols, matrixCol, output[row * SliceCols + col], matrix[matrixRow*Cols + matrixCol]);
  }
}

template<typename ElemT>
void matrixSlice(const uint Rows, const uint Cols, const ElemT* matrix, 
                 const uint startRow, const uint startCol, 
                 const uint SliceRows, const uint SliceCols,
                 ElemT* output, cudaStream_t stream, uint g, uint io, bool canPrint = false) {
  dim3 block = {256, 1, 1};
  dim3 grid = {SliceRows, 1, 1};
  //TODO: Make tile in column dimension also
  matrixSliceKernel<ElemT, 1, 1, 256><<<grid, block, 0, stream>>>
                                            (Rows, Cols, matrix,
                                             startRow, startCol, SliceRows, SliceCols,
                                             output);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename ElemT, typename VecT, bool K_EQUALS_VAR, uint VecTNumElems>
__device__ __forceinline__ 
void shiftAgToAsh(const bool RowsCModTileIsZero, const uint TileSizeRowsA, 
                  const uint TileSizeColsA, const uint MaxKronRows,
                  const uint TileSizeKronRows, const uint MaxColsA,
                  const uint NumThreads, const uint CRegRows,
                  const uint RowsC, const uint kronCols, const uint colsA,
                  const uint tid, const uint tileKronRow, const uint tileRowA,
                  const uint tile_k, const uint external_tile_kp_k,
                  const ElemT* __restrict__ glA, ElemT* __restrict__ shA) {
  // if (threadIdx.x == 0) printf("TileSizeRowsA %d\n", TileSizeRowsA);
  for (uint rowA = 0; rowA < (RowsCModTileIsZero ? TileSizeRowsA : MIN(TileSizeRowsA, RowsC - tileRowA)); rowA += 1) {
    for (uint a_col = tid*VecTNumElems; a_col < TileSizeColsA; a_col += NumThreads*VecTNumElems) {
      const ElemT* addrA;
      VecT  vec;
      ElemT elems[VecTNumElems];

      if (TileSizeKronRows == MaxKronRows) {
        addrA = &glA[(rowA + tileRowA) * colsA + (K_EQUALS_VAR ? 0 : tile_k*MaxColsA) + a_col];
        // *(VecT*)&shA[rowA][a_col] = a;
        // ElemT a1[4] = {a.x, a.y, a.z, a.w};
        // for (int j = 0; j < VecTNumElems; j++) {
        //   shA[rowA][a_col + j] = a1[j];
        // }
      } else {
        addrA = &glA[(rowA + tileRowA) * colsA + (K_EQUALS_VAR ? 0 : tile_k*MaxColsA) + \
                     (a_col/TileSizeKronRows)*kronCols + external_tile_kp_k * TileSizeKronRows + tileKronRow + a_col % TileSizeKronRows];
        // *(VecT*)&shA[rowA][a_col] = a;
      }

      globalLoadVec(addrA, vec);
      loadVecToRegs(vec, elems);

      #pragma unroll
      for (uint i = 0; i < VecTNumElems; i++) {
        uint ash_col = a_col + i;
        uint tileColA = (ash_col/TileSizeKronRows)/CRegRows; //(0,...,1024)/8 = (0,0,0,0,0,0 ... 127,127,127,127,127,127)
       
        uint final_col = (ash_col/TileSizeKronRows)*TileSizeKronRows + 
                         (tileColA + ash_col%TileSizeKronRows)%TileSizeKronRows;
        // if (blockIdx.x == 0&& blockIdx.y==0&&a_col < 64 && tileRowA == 0){
        //   printf("a_col %d final_col %d elem %f\n", a_col, rowA * TileSizeColsA +  final_col, elems[i]);
        // }
        // if (a_col +i < 8 && blockIdx.x == 0 && blockIdx.y == 0)
        //   printf("rowA %d a_col %d final_col %d TileSizeColsA %d elem %f\n", rowA, a_col + i, final_col, TileSizeColsA, elems[i]);
        shA[rowA * TileSizeColsA + final_col] = elems[i];
      }
    }
  }
}


template<typename ElemT, typename VecT, uint VecTNumElems>
__device__ __forceinline__ 
void tiledDirectFglToFsh(const uint MaxKronRows, const uint MaxKronCols, 
                         const uint TileSizeKronRows, const uint TileSizeKronCols,
                         const uint NumThreads, const uint external_tile_kp_n, const uint external_tile_kp_k, const uint tileKronRow, const uint kronRows, const uint kronCols, const uint tid, const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const uint loadInstr = MIN(TileSizeKronCols, VecTNumElems);
  //Create kronCols subwarps and each subwarp loads 0 to TileSizeKronRows elements
  for (uint swid = tid/(TileSizeKronCols/loadInstr); swid < TileSizeKronRows; swid += NumThreads/(TileSizeKronCols/loadInstr)) {
    VecT  vec;
    ElemT elems[VecTNumElems];

    const uint col = external_tile_kp_n*TileSizeKronCols + (tid%(TileSizeKronCols/loadInstr))*loadInstr;
    const uint row = swid;
    // shKronMats[tid%TileSizeKronRows][row] = glKronMats[(external_tile_kp_k * TileSizeKronCols + tileKronRow + row) * kronRows + col];

    globalLoadVec(&Fgl[(external_tile_kp_k * TileSizeKronRows + tileKronRow + row) * kronCols + col], vec);
    loadVecToRegs(vec, elems);

    #pragma unroll
    for (uint e = 0; e < loadInstr; e++) {
      uint linearIdx = (tid%(TileSizeKronCols/loadInstr))*loadInstr + e;
      Fsh[row * TileSizeKronCols + linearIdx] = elems[e];
    }
  }
}

template<typename ElemT, typename VecT, uint VecTNumElems>
__device__ __forceinline__ 
void fullDirectFglToFsh(const uint MaxKronRows, const uint MaxKronCols, 
                        const uint TileSizeKronRows, const uint TileSizeKronCols, 
                        const uint NumThreads, const uint kronRows, const uint kronCols, 
                        const uint tid, const ElemT* __restrict__ Fgl, ElemT* Fsh) {
  const uint loadInstr = MIN(kronRows*kronCols, VecTNumElems);

  for (uint eIdx = tid*loadInstr; eIdx < kronRows*kronCols; eIdx += blockDim.x*loadInstr) {
    ElemT regElems[VecTNumElems];
    VecT vec;

    vec = *(VecT*)&Fgl[eIdx];
    loadVecToRegs(vec, regElems);

    #pragma unroll
    for (uint vecElem = 0; vecElem < loadInstr; vecElem++) {
      uint idx = eIdx + vecElem;
      Fsh[(idx/MaxKronCols) * TileSizeKronCols + idx%MaxKronCols] = regElems[vecElem];
    }
  }
}

template<typename ElemT, uint NumFusedKerns>
struct KernelParams {
  const uint RowsC;
  const uint ColsC; //TODO: Change to LocalColsC
  const uint ColsA;
  uint KronRows[NumFusedKerns];
  uint KronCols[NumFusedKerns];
  const ElemT * __restrict__ glA;
  const ElemT * __restrict__ glKronMats[NumFusedKerns];
  ElemT       * __restrict__ glC;
  const uint kp_idx;

  KernelParams(const uint RowsC, const uint ColsC, const uint ColsA, const uint KronRows[NumFusedKerns],
               const uint KronCols[NumFusedKerns], const ElemT* glA,
               ElemT* glKronMats[NumFusedKerns], ElemT* glC, uint kp_idx) :
               RowsC(RowsC), ColsC(ColsC), ColsA(ColsA), glA(glA), glC(glC), kp_idx(kp_idx) {
    for (int i = 0; i < NumFusedKerns; i++) {
      this->KronRows[i] = KronRows[i];
      this->KronCols[i] = KronCols[i];
      this->glKronMats[i] = glKronMats[i];
    }
  }
};

const uint MaxGPUs = 8;
template<typename ElemT>
struct DistributedParams {
  ElemT* gpuResults1;
  ElemT* gpuResults2; //[MaxGPUs];
  const uint gr, gc;
  const uint numGPUs;
  const uint ColsA;
  const uint ColsC;
  const bool storeToDistMems;
  const uint LocalKrons;

  DistributedParams(ElemT* gpuResults1_, ElemT* gpuResults2_, const uint gr_, const uint gc_, const uint numGPUs_,   
                    const uint ColsA_, const uint ColsC_, const uint LocalKrons_, bool storeToDistMems_) :
    storeToDistMems(storeToDistMems_), gr(gr_), gc(gc_), numGPUs(numGPUs_), ColsA(ColsA_), ColsC(ColsC_),
    LocalKrons(LocalKrons_) {
      gpuResults1 = gpuResults1_;
      gpuResults2 = gpuResults2_;
    // assert (numGPUs_ < MaxGPUs);
    // for (int g = 0; g < numGPUs_; g++) {
    //   gpuResults[g] = gpuResults_[g];
    // }
    // for (int g = numGPUs_; g < MaxGPUs; g++) {
    //   gpuResults[g] = nullptr;
    // }
  }

  // DistributedParams(const DistributedParams<ElemT, LocalKrons>& x): numGPUs(x.numGPUs),
  //   ColsA(x.ColsA), ColsC(ColsC), storeToDistMems(storeToDistMems) {}

  //   DistributedParams<ElemT, LocalKrons>& operator=(const DistributedParams<ElemT, LocalKrons>& x) {

  //   }
};