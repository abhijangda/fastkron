#include "kmm/matrix.h"
#include "kernels/cuda/register-loads.cuh"

template<bool kXshSlicesSame, bool kPMultipleOfTileP, uint32_t TileP,
         typename ElemT, typename VecT, fastKronOp OpX, typename XShared>
CUDA_DEVICE
void shiftXgToXsh(const uint NumThreads, const uint RegK,
                  const uint tileP, const uint tid, const Slice<ElemT, OpX> XTile,
                  XShared& Xsh) {
  const int VecTLen = sizeof(VecT)/sizeof(ElemT);
  if (OpX == fastKronOp_N) {
    for (uint row = 0; row < XTile.m(); row += 1) {
    //Use NumThreads in the loop adder instead of blockDim.x for better perf
    for (uint k = tid*VecTLen; k < Xsh.n(); k += NumThreads*VecTLen) {
      ElemT regs[VecTLen] = {0};

      if (kPMultipleOfTileP && kXshSlicesSame) {
        ldGlobalVec(XTile.data(row, k, tileP), regs, VecTLen);
        Xsh.store(row, k, RegK, VecTLen, regs);
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
    }}
  } else if (OpX == fastKronOp_T) {
    //TODO: Similar to directFgToFsh. combine both?
    const uint Vecs     = XTile.m()/VecTLen;
    const uint ThGroups = MAX(1, NumThreads/Vecs);

    for (uint swid = tid/Vecs; swid < Xsh.n(); swid += ThGroups) {
    for (uint elem = tid%Vecs; elem < Vecs;    elem += NumThreads/ThGroups) {
      ElemT regs[VecTLen] = {0};

      const uint row = elem*VecTLen;
      const uint k = swid;
      if (kPMultipleOfTileP && kXshSlicesSame) {
        ldGlobalVec(XTile.data(row, k, tileP), regs, VecTLen);
        Xsh.store(row, k, RegK, VecTLen, regs);
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
          
          Fsh.store(row, elem * VecTLen, VecTLen, regs);
        } else if (opF == fastKronOp_T) {
          const uint row = tileQ*Fsh.q() + swid;
          const uint col = elem*VecTLen;

          if ((kPMultipleOfTileP || tileP + col < F.p()) &&
              (kQMultipleOfTileQ || row < F.q()))
            ldGlobalVec(F.data<ElemT>(tileP + col, row, opF), regs, VecTLen);
          
          for (int ii = 0; ii < VecTLen; ii++) {
            Fsh.store(elem * VecTLen + ii, swid, 1, &regs[ii]);
          }
        }

        //This condition avoids generating this loop giving better performance
        if (Vecs == NumThreads/ThGroups) break;
      }
    }
  } else if (Fsh.layout() == fastKronOp_T && !loadFullFactor) {
    const uint Vecs = Fsh.p()/VecTLen;
    const uint ThGroups = MAX(1, NumThreads/Vecs); //128/8 = 16

    for (uint swid = tid/Vecs; swid < Fsh.q(); swid += ThGroups) {
      for (uint elem = tid%Vecs; elem < Vecs; elem += NumThreads/ThGroups) {
        ElemT regs[VecTLen] = {0};
        if (opF == fastKronOp_T) {
          const uint col = elem*VecTLen;
          const uint row = tileQ*Fsh.q() + swid;

          // if ((kQMultipleOfTileQ || col < F.q()) &&
          //     (kPMultipleOfTileP || tileP + row < F.p()))
            ldGlobalVec(F.data<ElemT>(tileP + col, row, opF), regs, VecTLen);
          uint32_t shift = elem; //TODO: RegQ is 16
          //TODO: Consider this an array of TileP * TileQ. Do not worry about this being a fastKronOp_T layout.
          //Maybe we will not need this when FVecT = 4 because then number of bank conflicts decreases significantly.
          // if (false) //Shift
          //   (&Fsh.at(0,0))[elem * Fsh.q() + (shift + row)%Fsh.q()] = regs[0];//store(row, elemFsh.p(), VecTLen, regs);
          if (true) {//Padding
            #pragma unroll
            for (int ii = 0; ii < VecTLen; ii++) {
              (&Fsh.at(0,0))[(col+ii)*(Fsh.q()+1) + swid] = regs[ii];
            }
          }
          if (Vecs == NumThreads/ThGroups) break;
        }
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
        
        Xsh.store(tm, shXk, Yr.k(), 1, &Yr.at(tm, tk, tq));
  }}}}
}