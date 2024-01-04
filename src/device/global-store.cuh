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