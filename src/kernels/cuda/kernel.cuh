#include "config.h"

#include "kernels/params.h"

#include "kernels/cuda/utils.cuh"
#include "kernels/cuda/global-store.cuh"
#include "kernels/cuda/fixed-shape-tensor.cuh"
#include "kernels/cuda/mma.cuh"
#include "kernels/cuda/shared-loads.cuh"
#include "kernels/cuda/register-loads.cuh"

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
         uint MaxQ, uint MaxP, uint TileP, uint TileQ, uint TileK,
         uint TileM, uint FusedFacs, bool DistributeToGPUs, 
         uint RegK, uint RegQ,
         uint FactorHasMaxShape, 
         int XAlignment, int FAlignment,
         fastKronOp OpX, fastKronOp OpF>
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

  const Matrix X = params.problem.x();
  const Matrix Y = params.problem.y();

  const uint Q = (FactorHasMaxShape) ? MaxQ : params.problem.f(0).q();
  const uint P = (FactorHasMaxShape) ? MaxP : params.problem.f(0).p();

  //TODO: Make this Coord2D
  const uint tileQ = getTileQ<MaxQ, TileQ>();
  const uint tileK = getTileK<MaxQ, TileQ>();

  const uint tid      = threadIdx.x;
  const uint QThreads = (TileK / MaxP)     / RegK;
  const uint yQ       = (tid   / QThreads) * RegQ;
  const uint yK       = (tid   % QThreads) * RegK;
  
  const YElem yElem(yQ, yK);

  bool isThreadValid = (yElem.q() + RegQ <= TileQ);

  const uint tileM = blockIdx.y* TileM;

  Slice<ElemT, OpX> XTile(tileM, tileK * TileK, 
                          (TileM == 1) ? 1 : MIN(TileM, X.m() - tileM), TileK,
                          P, TileP,
                          X);
  
  extern __shared__ ElemT sharedStorage[];//[TileM*ShTileK + TileP*TileQ];
  
  using XShared = ShiftShared<fastKronOp_N, ElemT, TileM, ShTileK>;
  using FShared = DirectShared<OpF, ElemT, TileP, TileQ>;
  
  XShared Xsh(&sharedStorage[0]);
  FShared Fsh(&sharedStorage[Xsh.numel()]);

  register YRegisters<ElemT, TileM, RegK, RegQ> yReg;

  for (uint32_t tileP = 0; tileP < P; tileP += TileP) {
    //Loop iterates only once when FusedFacs == 1
    //Load X to shared memory
    shiftXgToXsh<ElemT, XVecT, OpX, decltype(Xsh)>(TileP, NumThreads, RegK,
                                              tileP, tid, XTile, Xsh);
    #pragma unroll
    for (int fac = FusedFacs - 1; fac >= 0; fac--) {
      const Factor F(P, Q, params.problem.f(fac).data());

      //Load F to shared memory
      directFgToFsh<ElemT, FVecT, decltype(Fsh)>(NumThreads, tid, OpF, tileP, tileQ,
                                                 F, Fsh);

      __syncthreads();

      //Zero out register results for fusion iterations
      if (FusedFacs > 1) yReg.zero();

      if (isThreadValid) {
        register XRegisters<ElemT, TileM, RegK, TileP> Xr;
        register FRegisters<ElemT, TileP, RegQ> Fr;

        mainMMA(XTile.m(), Xsh, Fsh, yReg, Xr, Fr, yElem);
      }

      if (FusedFacs > 1 && fac > 0) {
        __syncthreads();
        if (isThreadValid) {
          //Store C to shared memory using shift method
          fusionYrToXSh(XTile.m(), F, Fsh, Xsh, yReg, yElem);
        }
      }

      __syncthreads();
  }}

  if (!isThreadValid) return;

  #pragma unroll
  for (uint rm = 0; rm < yReg.m(); rm++) {
  if (rm < XTile.m()) {
    constexpr uint32_t StLen = storeVectorLen<FusedFacs, XAlignment, RegK>();
    #pragma unroll
    for (uint tq = 0; tq < RegQ; tq++) {
    #pragma unroll
    for (uint tk = 0; tk < RegK; tk += StLen) {
      const uint glM = rm + tileM;
      const uint32_t XTileSlices = TileK/P;
      //Total elements produced from TileK are (TileK/P) * Q
      //No. of elems produced by slice-multiply of TileK with 
      //the same col of F are: TileK/P, i.e, XTileSlices.
      //These elems are stored consecutively.

      //Compute element location inside the tile
      const uint32_t shK = (yElem.q()   + tq) * // F's col multiplied by this thread
                            XTileSlices +       // Index of first element produced by this F's col
                            yElem.k()   + tk ;  // index of element produced by multiplying this col with this slice
      uint glK;
      ElemT* outputArray;
      uint32_t cIdx;

      if (FusedFacs > 1) {
        glK = fusedYColumn(fusedParams, Y, Xsh, tileK, P, Q, shK);
      } else {
        //# of slices for a row. Same as X.n()/P but use Y.n()/Q to reduce
        //number of loads as store also requires reading Y.n()
        uint32_t XSlices = (Y.n()/Q);
        //Scale element location from within tile to global
        glK = (shK/XTileSlices)   * //The index of XTileSlices elems in TileK
              XSlices             + //Scale the index to global column
              tileK * XTileSlices + //Index of XTileSlices elems produced by a tileK 
              shK % XTileSlices;    //The element index within consecutive elems
        if (TileQ != Q) {
          const uint32_t tileQ     = getTileQ<MaxQ, TileQ>();
          const uint32_t NumQTiles = Q/TileQ;

          glK += tileQ*(Y.n()/NumQTiles);
      }}

      if (DistributeToGPUs) {
        outputArray = p2pStoreAddress<ElemT, DistributedParams>(distParams, Y, glM, glK);
      } else {
        cIdx = glM * Y.n() + glK;
        outputArray = (ElemT*)params.problem.y().data() + cIdx;
        if (params.kp_idx == 0) {
          #pragma unroll
          for (int i = 0; i < StLen; i++) {
            yReg.set(rm, tk+i, tq,
              epilogue(epilogueParams, cIdx + i, yReg.at(rm, tk + i, tq)));
      }}}

      stVecYReg(outputArray, yReg, StLen, rm, tk, tq);
  }}}}
}