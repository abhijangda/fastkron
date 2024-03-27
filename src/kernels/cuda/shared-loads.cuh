#include "kmm/matrix.h"
#include "kernels/cuda/register-loads.cuh"

template<typename ElemT, typename VecT, fastKronOp OpX, typename XShared>
CUDA_DEVICE
void shiftXgToXsh(const uint TileP, const uint NumThreads, const uint RegK,
                  const uint tileP, const uint tid, const Slice<ElemT, OpX> XTile,
                  XShared& Xsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  if (OpX == fastKronOp_N) {
    for (uint row = 0; row < XTile.m(); row += 1) {
    //Use NumThreads in the loop adder instead of blockDim.x for better perf
    for (uint k = tid*VecTLen; k < Xsh.n(); k += NumThreads*VecTLen) {
      ElemT regs[VecTLen];

      ldGlobalVec(XTile.data(row, k, tileP), regs, VecTLen);
      Xsh.store(row, k, TileP, RegK, VecTLen, regs);
    }}
  } else if (OpX == fastKronOp_T) {
    //TODO: Similar to directFgToFsh. combine both?
    const uint Vecs     = XTile.m()/VecTLen;
    const uint ThGroups = MAX(1, NumThreads/Vecs);

    for (uint swid = tid/Vecs; swid < Xsh.n(); swid += ThGroups) {
    for (uint elem = tid%Vecs; elem < Vecs;    elem += NumThreads/ThGroups) {
      ElemT regs[VecTLen];

      const uint row = elem*VecTLen;
      const uint k = swid;

      ldGlobalVec(XTile.data(row, k, tileP), regs, VecTLen);
      Xsh.store(row, k, TileP, RegK, VecTLen, regs);
    }}
  }
}

template<bool kExactShapes, typename ElemT, typename VecT, typename FShared>
CUDA_DEVICE
void directFgToFsh(const uint NumThreads, const uint tid, fastKronOp opF, 
                   const uint tileP, const uint tileQ,
                   const Factor& F, FShared& Fsh) {
  const uint VecTLen = sizeof(VecT)/sizeof(ElemT);

  if (!kExactShapes && (F.p() - tileP < Fsh.p() || F.q() < Fsh.q())) {
    //Zero out Fsh when remaining P is not equal to Fsh.p
    for (uint e = tid; e < Fsh.numel(); e += NumThreads) {
      ElemT regs[1] = {0};
      Fsh.store(e, 1, regs);
    }
    __syncthreads();
  }

  if (!(F.p() == Fsh.p() && F.q() == Fsh.q())) {
    //Create Fsh.p() thread groups and each group loads 0 to Fsh.q() elements
    const uint Vecs     = MIN(F.q(), Fsh.shape(1))/VecTLen;
    const uint ThGroups = MAX(1, NumThreads/Vecs);
    for (uint swid = tid/Vecs; swid < MIN(F.p(), Fsh.shape(0)); swid += ThGroups) {
      for (uint elem = tid%Vecs; elem < Vecs; elem += blockDim.x/ThGroups) {
        ElemT regs[VecTLen];
        if (opF == fastKronOp_N) {
          const uint col = tileQ*Fsh.q() + elem*VecTLen;
          const uint row = swid;

          ldGlobalVec(F.data<ElemT>(tileP + row, col, opF), regs, VecTLen);
        } else if (opF == fastKronOp_T) {
          const uint row = tileQ*Fsh.q() + swid;
          const uint col = elem*VecTLen;

          ldGlobalVec(F.data<ElemT>(tileP + col, row, opF), regs, VecTLen);
        }

        Fsh.store(swid, elem * VecTLen, VecTLen, regs);

        //This condition avoids generating this loop giving better performance
        if (Vecs == NumThreads/ThGroups) break;
      }
    }
  } else {
    //Optimized to load full factor matrix
    //Use blockDim in loop adder instead of NumThreads for better perf 
    for (uint eIdx = tid*VecTLen; eIdx < F.numel(); eIdx += blockDim.x*VecTLen) {
      ElemT regs[VecTLen];

      ldGlobalVec(F.data<ElemT>(eIdx), regs, VecTLen);
      Fsh.store(eIdx, VecTLen, regs);
}}}

template<typename FShared, typename XShared, typename YReg>
CUDA_DEVICE
void fusionYrToXSh(const uint32_t m, const Factor& F, const FShared& Fsh, XShared& Xsh, YReg& Yr, const YElem& yElem) {
  for (int tm = 0; tm < Yr.m(); tm++) {
    if (tm < m) {
      #pragma unroll
      for (uint tk = 0; tk < Yr.k(); tk++) {
      for (uint tq = 0; tq < Yr.q(); tq++) {
        const uint32_t MaxXSlices = Xsh.n()/F.p();
        uint32_t shXk = yElem.q()*MaxXSlices + tq*MaxXSlices + yElem.k() + tk;
        
        Xsh.store(tm, shXk, Fsh.p(), Yr.k(), 1, &Yr.at(tm, tk, tq));
  }}}}
}