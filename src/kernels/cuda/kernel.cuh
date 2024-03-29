#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/amd_detail/host_defines.h>
#endif

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

CUDA_DEVICE uint32_t getTileK(uint QByTileQ) {
  return blockIdx.x/QByTileQ;
}

CUDA_DEVICE uint32_t getTileQ(uint QByTileQ) {
  return blockIdx.x%QByTileQ;
}

template<uint kExactShapes, uint kTileK, uint kP, typename KernelParams> 
CUDA_DEVICE uint32_t getXshSlices(const KernelParams& params) {
  if (kExactShapes) {
    return kTileK/kP;
  } else {
    return params.XshSlices;
  }
}

template<uint kExactShapes, uint kQ, typename KernelParams> 
CUDA_DEVICE uint32_t getXSlices(const Matrix& Y, const KernelParams& params) {
  //# of slices for a row. Same as X.n()/P but use Y.n()/Q to reduce
  //number of loads as store also requires reading Y.n()
  if (kExactShapes) {
    return Y.n()/kQ;
  } else {
    return params.XSlices;
  }
}

template<uint kXshSlicesSame, uint RegK> 
CUDA_DEVICE uint32_t getQThreads(uint XshSlices) {
  if (kXshSlicesSame) return XshSlices/RegK;
  return DIVUP(XshSlices, RegK);
}

template<uint kQLeTileQ, uint TileQ> 
CUDA_DEVICE uint32_t getQByTileQ(uint Q) {
  if (kQLeTileQ) {
    return 1;
  }
  return DIVUP(Q, TileQ);
}

template<typename ElemT, typename Vec2T, typename Vec4T,
         uint NumThreads,
         uint MaxQ, uint MaxP, uint TileP, uint TileQ, uint kTileK,
         uint TileM, uint FusedFacs, bool DistributeToGPUs,
         uint RegK, uint RegQ,
         uint kExactShapes_,
         int XAlignment, int FAlignment,
         fastKronOp OpX, fastKronOp OpF>
__launch_bounds__(NumThreads)
__global__ void cudaKernel(KernelParams<FusedFacs> params,
                           FusedParams<FusedFacs> fusedParams,
                           DistributedParams distParams,
                           EpilogueParams epilogueParams) {
  //Alignment of X and F are correct in terms of elements of 4-bytes
  static_assert(XAlignment == 1 ||
                XAlignment == 2 ||
                XAlignment == 4,
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
  // static_assert(TileK % MaxP == 0,
  //               "TileK is not a multiple of MaxP");
  // static_assert((TileK/MaxP)%RegK == 0,
  //               "RegK not a multiple of MaxCols/MaxP");

  //Vector Load types based on alignments 
  using XVecT = ElemT;
                // typename std::conditional<XAlignment == 1, ElemT, 
                // typename std::conditional<XAlignment == 2, Vec2T, 
                //                           Vec4T>::type>::type;

  const bool LoadFullFactor = TileP >= MaxP && TileQ >= MaxQ && (MaxP*MaxQ) % 4 == 0;
  using FVecT = ElemT; 
                // typename std::conditional<LoadFullFactor , Vec4T,
                // typename std::conditional<FAlignment == 1, ElemT,
                // typename std::conditional<FAlignment == 2, Vec2T,
                //                           Vec4T>::type>::type>::type;

  const Matrix X = params.problem.x();
  const Matrix Y = params.problem.y();
  const bool kExactShapes = false;
  const uint Q = (kExactShapes) ? MaxQ : params.problem.f(0).q();
  const uint P = (kExactShapes) ? MaxP : params.problem.f(0).p();
  const bool kXshSlicesSame = false;
  const bool kQMultipleOfTileQ = false;
  const bool kPMultipleOfTileP = false;
  const bool kTileKMultipleOfK = false;
  const bool kQLeTileQ = false;
  const bool kTileKSame = false;

  const uint XshSlices = getXshSlices<kExactShapes, kTileK, MaxP>(params);
  const uint XSlices   = getXSlices  <kExactShapes, MaxQ>(Y, params);
  const uint QThreads  = getQThreads <kExactShapes || kXshSlicesSame, RegK>(XshSlices);
  const uint QByTileQ  = getQByTileQ <kQLeTileQ, TileQ>(Q);
  const uint TileK     = (kTileKSame) ? kTileK : params.tileX.n();
  
  const uint ShTileK   = XshSlices*TileP;

  //TODO: Make this Coord2D
  const uint tileQ = getTileQ(QByTileQ);
  const uint tileK = getTileK(QByTileQ);
  // if (threadIdx.x == 0) printf("Q %d tileQ %d blockIdx.x %d\n", Q, tileQ, blockIdx.x);

  const uint tid      = threadIdx.x;
  const uint yQ       = (tid   / QThreads) * RegQ;
  const uint yK       = (tid   % QThreads) * RegK;

  const YElem yElem(yQ, yK);

  const uint tileM = blockIdx.y * TileM;

  Slice<ElemT, OpX> XTile(tileM, tileK * TileK,
                          (TileM == 1) ? 1 : MIN(TileM, X.m() - tileM), TileK,
                          P, TileP,
                          X);

  extern __shared__ ElemT sharedStorage[];//[TileM*ShTileK + TileP*TileQ];

  using XShared = ShiftShared<fastKronOp_N, ElemT, kXshSlicesSame, 
                              TileM, kTileK/MaxP, TileP>;
  using FShared = DirectShared<OpF, ElemT, TileP, TileQ>;

  XShared Xsh(&sharedStorage[0], ShTileK);
  FShared Fsh(&sharedStorage[Xsh.numel()]);

  register YRegisters<ElemT, TileM, RegK, RegQ> yReg;

  for (uint32_t tileP = 0; tileP < P; tileP += TileP) {
    //Loop iterates only once when FusedFacs == 1
    //Load X to shared memory
    shiftXgToXsh<kXshSlicesSame, kPMultipleOfTileP, TileP, ElemT, XVecT, OpX, decltype(Xsh)>
                (NumThreads, RegK, tileP, tid, XTile, Xsh);
    #pragma unroll
    for (int fac = FusedFacs - 1; fac >= 0; fac--) {
      const Factor F(P, Q, params.problem.f(fac).data());

      //Load F to shared memory
      directFgToFsh<kPMultipleOfTileP, kQMultipleOfTileQ, ElemT, FVecT, decltype(Fsh)>
                    (NumThreads, tid, OpF, tileP, tileQ, F, Fsh);

      __syncthreads();

      //Zero out register results for fusion iterations
      if (FusedFacs > 1) yReg.zero();
      if (kExactShapes ||
          ((kTileKMultipleOfK || yElem.k() < MIN(XshSlices, XSlices - tileK * XshSlices)) &&
           (kQMultipleOfTileQ || yElem.q() < MIN(TileQ, Q - tileQ * TileQ))) //TODO: This is required only when XshSlices < 64
          ) {
        /*register*/ XRegisters<ElemT, TileM, RegK, TileP> Xr;
        /*register*/ FRegisters<ElemT, TileP, RegQ> Fr;

        mainMMA<kExactShapes>(XTile.m(), Xsh, Fsh, yReg, Xr, Fr, yElem);
      }

      if (FusedFacs > 1 && fac > 0) {
        __syncthreads();
        //Store C to shared memory using shift method
        fusionYrToXSh(XTile.m(), F, Fsh, Xsh, yReg, yElem);
      }

      __syncthreads();
  }}

  #pragma unroll
  for (uint rm = 0; rm < yReg.m(); rm++) {
  if (rm < XTile.m()) {
    constexpr uint32_t StLen = 1;//storeVectorLen<FusedFacs, XAlignment, RegK>();
    #pragma unroll
    for (uint tq = 0; tq < RegQ; tq++) {
    #pragma unroll
    for (uint tk = 0; tk < RegK; tk += StLen) {
      if ((!kTileKMultipleOfK && yElem.k() + tk >= MIN(XshSlices, XSlices - tileK * XshSlices)) || 
          (!kQMultipleOfTileQ && yElem.q() + tq >= MIN(TileQ, Q - tileQ * TileQ))) continue;

      const uint glM = rm + tileM;
      uint glK;
      ElemT* outputArray;
      uint32_t cIdx;

      //Total elements produced from TileK are (TileK/P) * Q
      //No. of elems produced by slice-multiply of TileK with
      //the same col of F are: TileK/P, i.e, XshSlices.
      //These elems are stored consecutively.
      if (FusedFacs > 1) {
        //Compute element location inside the tile
        const uint32_t shK = (yElem.q()   + tq) * // F's col multiplied by this thread
                              XshSlices +       // Index of first element produced by this F's col
                              yElem.k()   + tk ;  // index of element produced by multiplying this col with this slice
        glK = fusedYColumn(fusedParams, Y, Xsh, tileK, P, Q, shK);
      } else {
        //Scale element location from within tile to global
        glK = (yElem.q()   + tq)  * //The index of elems by one column in TileK
               XSlices            + //Scale the index to global column
               tileK * XshSlices  + //Index of XshSlices elems produced by a tileK 
               yElem.k()    + tk;   //The element index within consecutive elems
        if (TileQ < Q) {
          if (kExactShapes) {
            const uint32_t NumQTiles = Q/TileQ;
            glK += tileQ*(Y.n()/NumQTiles);
          } else {
            //TODO: This version is more general than previous version
            glK += tileQ * XSlices * TileQ;
          }
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
       }
      }
    }

      // if(glK >= Y.n()) printf("glK %d\n", glK);
      stVecYReg(outputArray, yReg, StLen, rm, tk, tq);
  }}}}
}