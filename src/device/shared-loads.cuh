#include "kmm/matrix.h"
#include "device/register-loads.cuh"

template<typename ElemT, typename VecT>
CUDA_DEVICE
void storeAgToAsh(const uint TileP, const uint NumThreads, const uint CRegRows,
                  const uint tid, const Slice<ElemT> XTile, ShiftShared& Xsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  for (uint row = 0; row < Xsh.m(); row += 1) {
    //Use NumThreads in loop adder instead of blockDim.x for better perf
    for (uint k = tid*VecTLen; k < Xsh.n(); k += NumThreads*VecTLen) {
      ElemT regs[VecTLen];

      ldGlobalVec((VecT*)XTile.data(row, k), regs);
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