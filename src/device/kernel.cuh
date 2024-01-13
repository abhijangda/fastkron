#include "config.h"

#include "device/utils.cuh"
#include "device/register-loads.cuh"
#include "device/shared-loads.cuh"
#include "device/params.h"
#include "device/mma.cuh"
#include "device/global-store.cuh"

#include <type_traits>
#include <typeinfo>

template<uint MaxQ, uint TileQ>
CUDA_DEVICE uint32_t getTileK() {
  return blockIdx.x/DIVUP(MaxQ, TileQ);
}

template<uint MaxQ, uint TileQ>
CUDA_DEVICE uint32_t getTileQ() {
  return blockIdx.x%DIVUP(MaxQ, TileQ);
}


template<typename ElemT, typename Vec2T, typename Vec4T,
         uint NumThreads, 
         uint MaxQ, uint MaxP, uint TileQ, uint TileK,
         uint TileM, uint FusedFacs, bool DistributeToGPUs, 
         uint RegK, uint RegQ,
         uint KPK_EQUALS_VAR, uint TileP, 
         int XAlignment, int FAlignment>
__launch_bounds__(NumThreads)
__global__ void kronGemmKernel(KernelParams<FusedFacs> params,
                               FusedParams<FusedFacs> fusedParams,
                               DistributedParams distParams,
                               EpilogueParams epilogueParams) {
  //Alignment of X and F are correct in terms of elements of 4-bytes
  static_assert(XAlignment    == 1 || 
                XAlignment    == 2 || 
                XAlignment    == 4,
                "Alignment of X should be 1, 2 or 4");
  static_assert(FAlignment == 1 || 
                FAlignment == 2 ||
                FAlignment == 4,
                "Alignment of Factor should be 1, 2 or 4");
  //Sanity Conditions on Tile Sizes
  static_assert(0 < TileQ && TileQ <= MaxQ, "");
  static_assert(FusedFacs == 1 ||
                (FusedFacs > 1 &&
                 TileP >= MaxP &&
                 TileQ >= MaxQ),
                "Invalid tile size params for fusion");
  static_assert(TileK % MaxP == 0,
                "TileK is not a multiple of MaxP");
  static_assert((TileK/MaxP)%RegK == 0,
                "RegK not a multiple of MaxCols/MaxP");
  
  //Vector Load types based on alignments 
  using XVecT = typename std::conditional<XAlignment == 1, ElemT, 
                typename std::conditional<XAlignment == 2, Vec2T, 
                                          Vec4T>::type>::type;
  
  const bool LoadFullFactor = TileP >= MaxP && TileQ >= MaxQ && (MaxP*MaxQ) % 4 == 0;
  using FVecT = typename std::conditional<LoadFullFactor , Vec4T,
                typename std::conditional<FAlignment == 1, ElemT,
                typename std::conditional<FAlignment == 2, Vec2T,
                                          Vec4T>::type>::type>::type;

  const uint ShTileK = TileK/(MaxP/TileP);

  __shared__ ElemT ptrXsh[TileM][ShTileK];
  __shared__ ElemT ptrFsh[TileP][TileQ];

  const Matrix X = params.problem.x();
  const Matrix Y = params.problem.y();

  const uint Q = (KPK_EQUALS_VAR) ? MaxQ : params.problem.f(0).q();
  const uint P = (KPK_EQUALS_VAR) ? MaxP : params.problem.f(0).p();

  const uint tileQ = getTileQ<MaxQ, TileQ>();
  const uint tileK = getTileK<MaxQ, TileQ>();

  const uint tid      = threadIdx.x;
  const uint QThreads = (TileK / MaxP)     / RegK;
  const uint yQ       = (tid   / QThreads) * RegQ;
  const uint yK       = (tid   % QThreads) * RegK;

  bool isThreadValid = (yQ + RegQ <= TileQ);

  const uint tileM = blockIdx.y* TileM;

  Slice<ElemT> XTile(tileM, tileK * TileK, 
                     (TileM == 1) ? 1 : MIN(TileM, X.m() - tileM), TileK,
                     P, TileP,
                     X);
  ShiftShared<ElemT> Xsh(XTile.m(), ShTileK, &ptrXsh[0][0]);
  DirectShared<Factor, ElemT> Fsh(TileP, TileQ, &ptrFsh[0][0], 0, tileQ);
  register YRegisters<ElemT, TileM, RegK, RegQ> yReg(yK, yQ);

  for (uint32_t tileP = 0; tileP < P; tileP += TileP) {
    //Loop iterates only once when FusedFacs == 1
    //Load X to shared memory
    shiftXgToXsh<ElemT, XVecT>(TileP, NumThreads, RegK,
                               tileP, tid, XTile, Xsh);

    #pragma unroll
    for (int fac = FusedFacs - 1; fac >= 0; fac--) {
      const Factor F(P, Q, params.problem.f(fac).data());
      
      //Load F to shared memory
      directFglToFsh<ElemT, FVecT>(NumThreads, tid, tileP, F, Fsh);
      
      __syncthreads();

      //Zero out register results for fusion iterations
      if (FusedFacs > 1) yReg.zero();
      
      if (isThreadValid) {
        register XRegisters<ElemT, TileM, RegK, TileP> Xr;
        register FRegisters<ElemT, TileP, RegQ> Fr;

        mainMMA<ElemT, decltype(Xsh), decltype(Fsh), decltype(yReg), 
                decltype(Xr), decltype(Fr)>(Xsh, Fsh, yReg, Xr, Fr);
      }
      
      if (FusedFacs > 1 && fac > 0) {
        __syncthreads();
        if (isThreadValid) {
          //Store C to shared memory using shift method
          fusionYrToXSh<ElemT, decltype(Xsh), decltype(yReg), TileP>(yQ, yK, F, Xsh, yReg);
        }
      }

      __syncthreads();
  }}

  if (!isThreadValid) return;

  #pragma unroll
  for (uint rowA = 0; rowA < yReg.m(); rowA++) {
  if (rowA < XTile.m()) {
    //TODO: Improve below code like in the paper
    constexpr uint32_t NumStElems = storeVectorElems<FusedFacs, XAlignment, RegK>();
    #pragma unroll
    for (uint reg_j = 0; reg_j < RegQ; reg_j++) {
    #pragma unroll
    for (uint reg_i = 0; reg_i < RegK; reg_i += NumStElems) {
      const uint cRow = (rowA + tileM);
      const uint32_t MaxXSlices = TileK/MaxP;
      uint shCol = yQ*MaxXSlices + reg_j*MaxXSlices + yK + reg_i;
      uint cCol = 0;
      ElemT* outputArray;
      uint32_t cIdx;

      if (FusedFacs > 1) {
        cCol = fusedYColumn<ElemT, decltype(fusedParams), decltype(Xsh)>(fusedParams, Y, Xsh, tileK, Q, shCol);
      } else {
        const uint32_t YSlices = Y.n()/Q;

        cCol = tileK * MaxXSlices + (shCol/MaxXSlices) * YSlices + shCol%MaxXSlices;
        if (TileQ != Q) {
          uint tileQ = getTileQ<MaxQ, TileQ>();
          const uint32_t NumQTiles = Q/TileQ;
          cCol += tileQ*(Y.n()/NumQTiles);
      }}

      if (DistributeToGPUs) {
        outputArray = p2pStoreAddress<ElemT, DistributedParams>(distParams, Y, cRow, cCol);
      } else {
        cIdx = cRow * Y.n() + cCol;
        outputArray = (ElemT*)params.problem.y().data() + cIdx;
        if (params.kp_idx == 0) {
          #pragma unroll
          for (int i = 0; i < NumStElems; i++) {
            yReg.regs[rowA][reg_i+i][reg_j] =
              epilogue(epilogueParams, cIdx + i, yReg.at(rowA,reg_i+i,reg_j));
      }}}

      stVecYReg<ElemT, decltype(yReg), NumStElems>(outputArray, yReg, rowA, reg_i, reg_j);
  }}}}
}