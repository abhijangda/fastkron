#include "common.h"

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