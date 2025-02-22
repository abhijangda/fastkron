#include "kmm/matrix.h"
#include "kernels/cuda/register-loads.cuh"

template<bool kMMultipleOfTileM, bool kXshSlicesSame, bool kPMultipleOfTileP, uint32_t TileP,
         typename ElemT, typename VecT, fastKronOp OpX, typename XShared>
CUDA_DEVICE
void shiftXgToXsh(const uint NumThreads, const uint RegK,
                  const uint tileP, const uint tid, const Slice<ElemT, OpX> XTile,
                  XShared& Xsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  if (OpX == fastKronOp_N) {
    const uint Vecs = Xsh.n()/VecTLen;
    const uint ThGroups = MAX(1, NumThreads/Vecs);

    for (uint row =  ((Xsh.layout() == fastKronOp_N) ? 0 : tid/Vecs);
              row <  XTile.m();
              row += ((Xsh.layout() == fastKronOp_N) ? 1 : ThGroups)) {
    //Use NumThreads in the loop adder instead of blockDim.x for better perf
    for (uint k =  ((Xsh.layout() == fastKronOp_N) ? tid * VecTLen : tid % Vecs);
              k <  ((Xsh.layout() == fastKronOp_N) ? Xsh.n() : Vecs);
              k += ((Xsh.layout() == fastKronOp_N) ? NumThreads * VecTLen : NumThreads/ThGroups)) {
      ElemT regs[VecTLen] = {0};
      const fastKronOp elemOp = (Xsh.layout() == fastKronOp_N) ?
                                  fastKronOp_N : //MKM
                                  fastKronOp_T; //KMM

      if (kPMultipleOfTileP && kXshSlicesSame) {
        if (Xsh.layout() == fastKronOp_N) {
          ldGlobalVec(XTile.data(row, k, tileP), regs, VecTLen);
          Xsh.store(row, k, RegK, VecTLen, regs, elemOp);
        } else {
          ldGlobalVec(XTile.data(row, k * VecTLen, tileP), regs, VecTLen);
          //When MMType is KMM then VecTLen is 1
          Xsh.store(row, k * VecTLen, RegK, VecTLen, regs, elemOp);
          // for (int i = 0; i < VecTLen; i++)
          //   Xsh.store(row, k * VecTLen + i, RegK, 1, &regs[i], elemOp);
        }
      } else {
        //TODO: Valid only when VecTLen == 1
        uint32_t slice = k/Xsh.p();
        uint32_t elem = k%Xsh.p();

        uint32_t xidx = XTile.data(row, slice, elem, tileP);
        if (kPMultipleOfTileP || (tileP + elem < XTile.P && slice < XTile.cols/XTile.P)) {
          ldGlobalVec(XTile.data(xidx), regs, VecTLen);  
        } else {
          //TODO: Remaining less than VecTLen elems
        }
        Xsh.store(row, slice, elem, RegK, VecTLen, regs);
      }

      //This condition avoids generating this loop giving better performance
      if (Xsh.layout() == fastKronOp_T && Vecs == NumThreads/ThGroups) break;
    }}
  } else if (OpX == fastKronOp_T) {
    //TODO: Similar to directFgToFsh. combine both?
    const uint Vecs     = Xsh.m()/VecTLen;
    const uint ThGroups = MAX(1, NumThreads/Vecs);

    for (uint swid = tid/Vecs; swid < Xsh.n(); swid += ThGroups) {
    for (uint elem = tid%Vecs;
         elem < Vecs && (kMMultipleOfTileM ? true : elem*VecTLen < XTile.m()); 
         elem += NumThreads/ThGroups) {
      ElemT regs[VecTLen] = {0};

      const uint row = elem*VecTLen;
      const uint k = swid;
      const fastKronOp elemOp = (Xsh.layout() == fastKronOp_N) ?
                                  fastKronOp_T : //MKM
                                  fastKronOp_N; //KMM
      if (kPMultipleOfTileP && kXshSlicesSame) {
        ldGlobalVec(XTile.data(row, k, tileP), regs, VecTLen);
        Xsh.store(row, k, RegK, VecTLen, regs, elemOp);
      } else {
        uint32_t slice = k/Xsh.p();
        uint32_t elem  = k%Xsh.p();

        uint32_t xidx = XTile.data(row, slice, elem, tileP);
        if (kPMultipleOfTileP || (tileP + elem < XTile.P && slice < XTile.cols/XTile.P)) {
          ldGlobalVec(XTile.data(xidx), regs, VecTLen);  
        } else {
          //TODO: Remaining less than VecTLen elems
        }
        Xsh.store(row, slice, elem, RegK, VecTLen, regs);
      }
    }}
  }
}

template<bool kPMultipleOfTileP, bool kQMultipleOfTileQ,
         typename ElemT, typename VecT, fastKronOp opF, typename FShared>
CUDA_DEVICE
void directFgToFsh(const uint NumThreads, const uint tid, 
                   const uint tileP, const uint tileQ,
                   const Factor& F, FShared& Fsh) {
  const uint VecTLen = sizeof(VecT)/sizeof(ElemT);

  if ((!kQMultipleOfTileQ && F.q() < Fsh.q()) || 
      (!kPMultipleOfTileP && (F.p() - tileP < Fsh.p()))) {
    //Zero out Fsh when remaining P is not equal to Fsh.p
    for (uint e = tid; e < Fsh.numel(); e += NumThreads) {
      ElemT regs[1] = {0};
      Fsh.store(e, 1, regs);
    }
    __syncthreads();
  }

  bool loadFullFactor = F.p() == Fsh.p() && F.q() == Fsh.q();

  if (Fsh.layout() == fastKronOp_N && (opF == fastKronOp_T || !loadFullFactor)) {
    //MKM
    //Create Fsh.p() thread groups and each group loads 0 to Fsh.q() elements
    const uint Vecs     = ((opF == fastKronOp_N) ? Fsh.q() : Fsh.p())/VecTLen;
    const uint ThGroups = MAX(1, NumThreads/Vecs);
    for (uint swid = tid/Vecs; swid < ((opF == fastKronOp_N) ? Fsh.p() : Fsh.q()); swid += ThGroups) {
      for (uint elem = tid%Vecs; elem < Vecs; elem += NumThreads/ThGroups) {
        ElemT regs[VecTLen] = {0};
        if (opF == fastKronOp_N) {
          const uint col = tileQ*Fsh.q() + elem*VecTLen;
          const uint row = swid;

          if ((kQMultipleOfTileQ || col < F.q()) &&
              (kPMultipleOfTileP || tileP + row < F.p()))
            ldGlobalVec(F.data<ElemT>(tileP + row, col, opF), regs, VecTLen);
          
          Fsh.store(row, elem * VecTLen, VecTLen, regs, fastKronOp_N);
        } else if (opF == fastKronOp_T) {
          const uint row = tileQ*Fsh.q() + swid;
          const uint col = elem*VecTLen;

          if ((kPMultipleOfTileP || tileP + col < F.p()) &&
              (kQMultipleOfTileQ || row < F.q()))
            ldGlobalVec(F.data<ElemT>(tileP + col, row, opF), regs, VecTLen);
          
          Fsh.store(elem * VecTLen, swid, VecTLen, regs, fastKronOp_N);
        }

        //This condition avoids generating this loop giving better performance
        if (Vecs == NumThreads/ThGroups) break;
      }
    }
  } else if (Fsh.layout() == fastKronOp_T) {
    //KMM
    const uint Vecs = ((opF == fastKronOp_N) ? Fsh.q() : Fsh.p())/VecTLen;
    const uint ThGroups = MAX(1, NumThreads/Vecs);
  
    for (uint swid = tid/Vecs; swid < ((opF == fastKronOp_N) ? Fsh.p() : Fsh.q()); swid += ThGroups) {
      for (uint elem = tid%Vecs; elem < Vecs; elem += NumThreads/ThGroups) {
        ElemT regs[VecTLen] = {0};
        if (opF == fastKronOp_T) {
          const uint col = elem*VecTLen;
          const uint row = tileQ*Fsh.q() + swid;

          if ((kQMultipleOfTileQ || row < F.q()) &&
              (kPMultipleOfTileP || tileP + col < F.p()))
            ldGlobalVec(F.data<ElemT>(tileP + col, row, opF), regs, VecTLen);

          Fsh.store(col, swid, VecTLen, regs, fastKronOp_T);
        } else if (opF == fastKronOp_N) {
          const uint col = tileQ*Fsh.q() + elem*VecTLen;
          const uint row = swid;

          if ((kQMultipleOfTileQ || col < F.q()) &&
              (kPMultipleOfTileP || tileP + row < F.p()))
            ldGlobalVec(F.data<ElemT>(tileP + row, col, opF), regs, VecTLen);

          Fsh.store(row, elem*VecTLen, VecTLen, regs, fastKronOp_N);
        }

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
  if (Xsh.layout() == fastKronOp_N) {
    //MKM
    for (int tm = 0; tm < Yr.m(); tm++) {
      if (true) {//TODO: probably do not need this condition
        #pragma unroll
        for (uint tk = 0; tk < Yr.k(); tk++) {
        for (uint tq = 0; tq < Yr.q(); tq++) {
          const uint32_t MaxXSlices = Xsh.n()/F.p();
          uint32_t shXk = yElem.q()*MaxXSlices + tq*MaxXSlices + yElem.k() + tk;
          
          Xsh.store(yElem.m() + tm, shXk, Yr.k(), 1, &Yr.at(tm, tk, tq), fastKronOp_N);
    }}}}
  } else if (Xsh.layout() == fastKronOp_T) {
    //KMM
    #pragma unroll
    for (uint tk = 0; tk < Yr.k(); tk++) {
    for (uint tq = 0; tq < Yr.q(); tq++) {
    #pragma unroll
    for (int tm = 0; tm < Yr.m(); tm++) {
      const uint32_t MaxXSlices = Xsh.n()/F.p();
      uint32_t shXk = yElem.q()*MaxXSlices + tq*MaxXSlices + yElem.k() + tk;
      
      Xsh.store(yElem.m() + tm, shXk, Yr.k(), 1, &Yr.at(tm, tk, tq), fastKronOp_N);
    }}}
  }
}