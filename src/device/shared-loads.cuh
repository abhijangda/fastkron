#include "kmm/matrix.h"
#include "device/register-loads.cuh"

template<typename ElemT, typename VecT>
CUDA_DEVICE
void shiftXgToXsh(const uint TileP, const uint NumThreads, const uint RegK,
                  const uint tileP, const uint tid, const Slice<ElemT> XTile,
                  ShiftShared<ElemT>& Xsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  for (uint row = 0; row < Xsh.m(); row += 1) {
    //Use NumThreads in the loop adder instead of blockDim.x for better perf
    for (uint k = tid*VecTLen; k < Xsh.n(); k += NumThreads*VecTLen) {
      ElemT regs[VecTLen];

      ldGlobalVec((VecT*)XTile.data(row, k, tileP), regs);
      Xsh.store(row, k, TileP, RegK, VecTLen, regs);
    }
  }
}

template<typename ElemT, typename VecT>
CUDA_DEVICE
void directFglToFsh(const uint NumThreads, const uint tid, const uint tileP,
                    const Factor& F, DirectShared<Factor, ElemT>& Fsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  
  if (!(F.p() == Fsh.p() && F.q() == Fsh.q())) {
    //Create Fsh.p() thread groups and each group loads 0 to Fsh.q() elements
    const uint QVecs    = Fsh.q()/VecTLen;
    const uint ThGroups = MAX(1, NumThreads/QVecs);

    for (uint swid = tid/QVecs; swid < Fsh.p(); swid += ThGroups) {
      for (uint qelem = tid%QVecs; qelem < QVecs; qelem += blockDim.x/ThGroups) {
        ElemT elems[VecTLen];

        const uint col = Fsh.tilecol*Fsh.q() + qelem*VecTLen;
        const uint row = swid;

        ldGlobalVec((VecT*)F.data<ElemT>((tileP + row), col), elems);
        Fsh.store(row, qelem * VecTLen, VecTLen, elems);

        //This condition avoids generating this loop giving better performance
        if (QVecs == NumThreads/ThGroups) break;
  }}} else {
    //Optimized to load full factor matrix
    //Use blockDim in loop adder instead of NumThreads for better perf 
    for (uint eIdx = tid*VecTLen; eIdx < F.numel(); eIdx += blockDim.x*VecTLen) {
      ElemT regs[VecTLen];

      ldGlobalVec((VecT*)F.data<ElemT>(eIdx), regs);
      Fsh.store(eIdx, VecTLen, regs);
}}}

template<typename FShared, typename XShared, typename YReg>
CUDA_DEVICE
void fusionYrToXSh(const Factor& F, const FShared& Fsh, XShared& Xsh, YReg& Yr) {
  for (int tm = 0; tm < Yr.m(); tm++) {
    if (tm < Xsh.m()) {
      #pragma unroll
      for (uint tk = 0; tk < Yr.k(); tk++) {
      for (uint tq = 0; tq < Yr.q(); tq++) {
        uint shXk = Yr.yQ*(Xsh.n()/F.p()) + tq*(Xsh.n()/F.p()) + Yr.yK + tk;
        
        Xsh.store(tm, shXk, Fsh.p(), Yr.k(), 1, &Yr.at(tm, tk, tq));
  }}}}
}