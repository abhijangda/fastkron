#include "kmm/matrix.h"
#include "device/register-loads.cuh"

template<typename ElemT, typename VecT, typename XShared>
CUDA_DEVICE
void shiftXgToXsh(const uint TileP, const uint NumThreads, const uint RegK,
                  const uint tileP, const uint tid, const Slice<ElemT> XTile,
                  XShared& Xsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  for (uint row = 0; row < XTile.m(); row += 1) {
    //Use NumThreads in the loop adder instead of blockDim.x for better perf
    for (uint k = tid*VecTLen; k < Xsh.n(); k += NumThreads*VecTLen) {
      ElemT regs[VecTLen];

      ldGlobalVec(XTile.data(row, k, tileP), regs, VecTLen);
      Xsh.store(row, k, TileP, RegK, VecTLen, regs);
    }
  }
}

template<typename ElemT, typename VecT, typename FShared>
CUDA_DEVICE
void directFgToFsh(const uint NumThreads, const uint tid, const uint tileP, const uint tileQ,
                   const Factor& F, FShared& Fsh) {
  const uint VecTLen = sizeof(VecT)/sizeof(ElemT);

  if (!(F.p() == Fsh.p() && F.q() == Fsh.q())) {
    //Create Fsh.p() thread groups and each group loads 0 to Fsh.q() elements
    const uint QVecs    = Fsh.q()/VecTLen;
    const uint ThGroups = MAX(1, NumThreads/QVecs);

    for (uint swid = tid/QVecs; swid < Fsh.p(); swid += ThGroups) {
      for (uint qelem = tid%QVecs; qelem < QVecs; qelem += blockDim.x/ThGroups) {
        ElemT regs[VecTLen];

        const uint col = tileQ*Fsh.q() + qelem*VecTLen;
        const uint row = swid;

        ldGlobalVec(F.data<ElemT>((tileP + row), col), regs, VecTLen);
        Fsh.store(row, qelem * VecTLen, VecTLen, regs);

        //This condition avoids generating this loop giving better performance
        if (QVecs == NumThreads/ThGroups) break;
  }}} else {
    //Optimized to load full factor matrix
    //Use blockDim in loop adder instead of NumThreads for better perf 
    for (uint eIdx = tid*VecTLen; eIdx < F.numel(); eIdx += blockDim.x*VecTLen) {
      ElemT regs[VecTLen];

      ldGlobalVec(F.data<ElemT>(eIdx), regs, VecTLen);
      Fsh.store(eIdx, VecTLen, regs);
}}}

template<typename FShared, typename XShared, typename YReg>
CUDA_DEVICE
void fusionYrToXSh(const uint32_t m, const Factor& F, const FShared& Fsh, XShared& Xsh, YReg& Yr) {
  for (int tm = 0; tm < Yr.m(); tm++) {
    if (tm < m) {
      #pragma unroll
      for (uint tk = 0; tk < Yr.k(); tk++) {
      for (uint tq = 0; tq < Yr.q(); tq++) {
        const uint32_t MaxXSlices = Xsh.n()/F.p();
        uint32_t shXk = Yr.yQ*MaxXSlices + tq*MaxXSlices + Yr.yK + tk;
        
        Xsh.store(tm, shXk, Fsh.p(), Yr.k(), 1, &Yr.at(tm, tk, tq));
  }}}}
}