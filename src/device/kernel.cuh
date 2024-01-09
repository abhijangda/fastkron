#include "config.h"

#include "device/utils.cuh"
#include "device/register-loads.cuh"
#include "device/shared-loads.cuh"
#include "device/params.h"
#include "device/mma.cuh"
#include "device/global-store.cuh"

#include <type_traits>
#include <typeinfo>

template<typename ElemT, typename Vec2T, typename Vec4T,
         uint NumThreads, 
         uint MaxQ, uint MaxP, uint TileQ, uint TileK,
         uint TileM, uint FusedMuls, bool DistributeToGPUs, 
         uint CRegRows, uint CRegCols,
         uint KPK_EQUALS_VAR, uint TileP, 
         int XAlignment, int FAlignment>
__launch_bounds__(NumThreads)
__global__ void kronGemmKernel(KernelParams<FusedMuls> params,
                               FusedParams<FusedMuls> fusedParams,
                               DistributedParams distParams,
                               EpilogueParams epilogueParams) {
  static_assert(XAlignment    == 1 || XAlignment    == 2 || XAlignment    == 4,
                "Alignment of X should be 1, 2 or 4");
  static_assert(FAlignment == 1 || FAlignment == 2 || FAlignment == 4,
                "Alignment of Factor should be 1, 2 or 4");
  static_assert(0 < TileQ && TileQ <= MaxQ, "");
  static_assert(FusedMuls == 1 ||
                (FusedMuls > 1 && TileP >= MaxP && TileQ >= MaxQ),
                "Invalid tile size params for fusion");
  static_assert(TileK % MaxP          == 0, "TileK is not a multiple of MaxP");
  static_assert((TileK/MaxP)%CRegRows == 0, "CRegRows not a multiple of MaxCols/MaxP");
  
  using XVecT = typename std::conditional<XAlignment == 1, ElemT, 
                typename std::conditional<XAlignment == 2, Vec2T, 
                                          Vec4T>::type>::type;
  using FVecT = typename std::conditional<TileP >= MaxP && TileQ >= MaxQ && (MaxP*MaxQ) % 4 == 0, Vec4T, //Load full factor using 4 elems
                typename std::conditional<FAlignment == 1, ElemT,
                typename std::conditional<FAlignment == 2, Vec2T,
                                          Vec4T>::type>::type>::type;

  const uint tid     = threadIdx.x;
  const uint ShTileK = TileK/(MaxP/TileP);

  __shared__ ElemT ptrXsh[TileM][ShTileK];
  __shared__ ElemT ptrFsh[TileP][TileQ];

  uint Q;
  uint P;

  if (KPK_EQUALS_VAR) {
    Q = MaxQ;
    P = MaxP;
  } else {
    Q = params.problem.f(0).q();
    P = params.problem.f(0).p();
  }

  const Matrix X = params.problem.x();
  const Matrix Y = params.problem.y();

  const uint tileQ = get_external_tile_kp_n<MaxQ, TileQ>();
  const uint MaxL = (TileK/MaxP)*MaxQ;
  constexpr uint wSz = (TileK/MaxP)/CRegRows;

  const uint kp_col_start_ = (tid / wSz) * CRegCols;
  const uint a_col_start_  = (tid % wSz) * CRegRows;

  const uint tileRowA         = blockIdx.y * TileM;
  const uint outerTileKronCol = kp_col_start_;
  const uint tileColC         = a_col_start_ ;
  
  bool isThreadValid = (kp_col_start_ + CRegCols <= TileQ);
  uint tileK = get_tile_k<MaxQ, TileQ>();

  Slice<ElemT> XTile(tileRowA, tileK * TileK, 
                     (TileM == 1) ? 1 : MIN(TileM, X.m() - tileRowA),
                     TileK, P, TileP, X);
  ShiftShared Xsh(XTile.m(), ShTileK, &ptrXsh[0][0]);
  DirectShared<Factor, ElemT> Fsh(TileP, TileQ, &ptrFsh[0][0], 0, tileQ);
  register YRegisters<ElemT, TileM, CRegRows, CRegCols> yReg;
  //TODO: Make tileP and remove it from XTile
  for ( ; XTile.valid(); XTile.nextTileP()) {
    //Loop iterates only once when FusedMuls == 1
    storeAgToAsh<ElemT, XVecT>(TileP, NumThreads, CRegRows,
                               tid, XTile, Xsh);

    #pragma unroll
    for (int fusedFac = FusedMuls - 1; fusedFac >= 0; fusedFac--) {
      if (FusedMuls > 1) {
        yReg.clear();
      }

      const ElemT* __restrict__ Fgl = (ElemT*)params.problem.f(fusedFac).data();
      const Factor F(P, Q, params.problem.f(fusedFac).data());

      directFglToFsh<ElemT, FVecT>(NumThreads, tid, XTile.tileP, F, Fsh);
      __syncthreads();

      if (isThreadValid) {
        mainMMA<ElemT, decltype(Xsh), decltype(Fsh), decltype(yReg), TileM, CRegRows, CRegCols, TileP>
          (tileColC, outerTileKronCol, Xsh, Fsh, yReg);
      }

      __syncthreads();

      if (isThreadValid && FusedMuls > 1 && fusedFac > 0) {
        //Store C to shared memory using shift method
        fusionYrToXSh<ElemT, decltype(Xsh), decltype(yReg), TileP>(outerTileKronCol, tileColC, F, Xsh, yReg);
      }
      __syncthreads();
    }
  }

  if (!isThreadValid) return;

  #pragma unroll
  for (uint rowA = 0; rowA < yReg.TileM(); rowA++) {
  if (rowA < XTile.m()) {
    //TODO: Improve below code like in the paper
    constexpr uint32_t stVecElems = storeVectorElems<FusedMuls, XAlignment, CRegRows>();
    #pragma unroll
    for (uint reg_j = 0; reg_j < CRegCols; reg_j++) {
    #pragma unroll
    for (uint reg_i = 0; reg_i < CRegRows; reg_i += stVecElems) {
      const uint cRow = (rowA + tileRowA);
      uint shCol = outerTileKronCol*(TileK/MaxP) + reg_j*(TileK/MaxP) + tileColC + reg_i;
      uint cCol = 0;
      ElemT* outputArray;
      uint32_t cIdx;

      if (FusedMuls > 1) {
        cCol = fusedYColumn<ElemT, decltype(fusedParams), decltype(Xsh)>(fusedParams, Y, Xsh, tileK, Q, shCol);
      } else {
        cCol = tileK * (MaxL/Q) + (shCol/(MaxL/Q)) * (Y.n()/Q) + shCol%(MaxL/Q);
      }

      if (TileQ != Q) {
        uint tileQ = get_external_tile_kp_n<MaxQ, TileQ>();
        cCol += tileQ*(Y.n()/(Q/TileQ)); 
      }

      if (DistributeToGPUs) {
        outputArray = p2pStoreAddress<ElemT, decltype(distParams)>(distParams, Y, cRow, cCol);
      } else {
        cIdx = cRow * Y.n() + cCol;
        outputArray = (ElemT*)params.problem.y().data() + cIdx;
        if (params.kp_idx == 0) {
          #pragma unroll
          for (int i = 0; i < stVecElems; i++) {
            yReg.regs[rowA][reg_i+i][reg_j] =
              epilogue(epilogueParams, cIdx + i, yReg.at(rowA,reg_i+i,reg_j));
          }
        }
      }

      register ElemT storeValues[stVecElems];
      #pragma unroll
      for (int v = 0; v < stVecElems; v++) {
        storeValues[v] = yReg.at(rowA, reg_i+v, reg_j);
      }

      stGlobalVec(outputArray, stVecElems, storeValues);
  }}}}
}