#include "kmm/matrix.h"
#include "device/register-loads.cuh"

template<typename ElemT, typename VecT>
CUDA_DEVICE
void storeAgToAsh(const uint TileP, const uint NumThreads, const uint CRegRows,
                  const uint tileP, const uint tid, const Slice<ElemT> XTile, ShiftShared& Xsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  for (uint row = 0; row < Xsh.m(); row += 1) {
    //Use NumThreads in loop adder instead of blockDim.x for better perf
    for (uint k = tid*VecTLen; k < Xsh.n(); k += NumThreads*VecTLen) {
      ElemT regs[VecTLen];

      ldGlobalVec((VecT*)XTile.data(row, k, tileP), regs);
      Xsh.store<ElemT, VecTLen>(row, k, TileP, CRegRows, regs);
    }
  }
}

template<typename ElemT, typename VecT>
CUDA_DEVICE
void directFglToFsh(const uint NumThreads,
                    const uint tid, const uint tileP,
                    const Factor& F, DirectShared<Factor, ElemT>& Fsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  
  if (!(F.p() == Fsh.p() && F.q() == Fsh.q())) {
    //Create F.q() subwarps and each subwarp loads 0 to TileP elements
    const uint QVecs = Fsh.q()/VecTLen;
    const uint subWarps = MAX(1, NumThreads/QVecs);
    for (uint swid = tid/QVecs; swid < Fsh.p(); swid += subWarps) {
      ElemT regs[VecTLen];

      for (uint elem = tid%QVecs; elem < QVecs; elem += blockDim.x/subWarps) {
        const uint col = Fsh.tilecol*Fsh.q() + elem*VecTLen;
        const uint row = swid;

        ldGlobalVec((VecT*)F.data<ElemT>((tileP + row), col), regs);

        Fsh.store(row, elem * VecTLen, VecTLen, regs);

        //This condition avoids generating the loop giving better performance
        if (QVecs == NumThreads/subWarps) break;
      }
    }
  } else {
    //Optimized to load full factor matrix
    //Use blockDim in loop adder instead of NumThreads for better perf 
    for (uint eIdx = tid*VecTLen; eIdx < F.numel(); eIdx += blockDim.x*VecTLen) {
      ElemT regs[VecTLen];

      ldGlobalVec((VecT*)F.data<ElemT>(eIdx), regs);

      Fsh.store(eIdx, VecTLen, regs);
    }
  }
}

template<typename ElemT, typename XShared, typename YReg, uint32_t TileP>
__device__
void fusionYrToXSh(uint32_t outerTileKronCol, uint32_t tileColC, const Factor& F, XShared& Xsh, YReg& Yr) {
  for (int rowA = 0; rowA < Yr.TileM(); rowA++) {
    if (rowA < Xsh.m()) {
      #pragma unroll
      for (uint reg_i = 0; reg_i < Yr.SliceM(); reg_i++) {
      for (uint reg_j = 0; reg_j < Yr.SliceN(); reg_j++) {
        uint cCol = outerTileKronCol*(Xsh.n()/F.p()) + reg_j*(Xsh.n()/F.p()) + tileColC + reg_i;
        uint tileColC_ = (cCol/TileP)/Yr.SliceM(); //TODO: This is shift?
        
        cCol = (cCol/TileP)*TileP + (tileColC_ + cCol%TileP)%TileP;
        Xsh.template set<ElemT>(rowA, cCol, Yr.at(rowA, reg_i, reg_j));
  }}}}
}