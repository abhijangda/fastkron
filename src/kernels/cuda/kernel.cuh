#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/amd_detail/host_defines.h>
#endif

#include "config.h"

#include "kernels/params.h"
#include "kernels/get_batched_data.h"

#include "kernels/cuda/utils.cuh"
#include "kernels/cuda/global-store.cuh"
#include "kernels/cuda/fixed-shape-tensor.cuh"
#include "kernels/cuda/mma.cuh"
#include "kernels/cuda/shared-loads.cuh"
#include "kernels/cuda/register-loads.cuh"

#include <type_traits>
#include <typeinfo>

CUDA_DEVICE uint32_t getTileK(uint bid_x, uint QByTileQ) {
  return bid_x/QByTileQ;
}

CUDA_DEVICE uint32_t getTileQ(uint bid_x, uint QByTileQ) {
  return bid_x%QByTileQ;
}

CUDA_DEVICE
YElem getYElem(const uint32_t tid, fastKronOp OpY, const uint32_t NumThreads, uint32_t QThreads,
                const uint32_t MaxP,
                const uint32_t TileM, const uint32_t kTileK, const uint32_t TileQ,
                  const uint32_t RegM,  const uint32_t RegK,   const uint32_t RegQ) {
  if (OpY == fastKronOp_N) {
    const uint MThreads  = (TileM == 1) ? NumThreads : (TileQ/RegQ) * ((kTileK/MaxP)/RegK);
    const uint yQ   = ((tid % MThreads) / QThreads) * RegQ;
    const uint yK   = ((tid % MThreads) % QThreads) * RegK;
    const uint yM   = (MThreads >= NumThreads) ? 0 : ((tid / MThreads) * RegM);
  
    return YElem(yM, yQ, yK);
  } else if (OpY == fastKronOp_T) {
    QThreads = QThreads * (TileM/RegM);
    const uint KThreads  = TileM/RegM;
    const uint yQ   = (tid / QThreads) * RegQ;
    const uint yM   = ((tid % QThreads) % KThreads) * RegM;
    const uint yK   = ((tid % QThreads) / KThreads) * RegK;

    return YElem(yM,yQ,yK);
  }
}

template<uint SMArch, typename ElemT, typename Vec2T, typename Vec4T,
         uint NumThreads,
         uint MaxQ, uint MaxP, uint TileP, uint TileQ, uint kTileK,
         uint TileM, uint FusedFacs, bool DistributeToGPUs,
         uint RegM, uint RegK, uint RegQ,
         uint OptLevel,
         int XAlignment, int FAlignment,
         fastKronOp kOpX, fastKronOp kOpF, FastKronMMType kmmType, KernelBatchType::Ty KernelBatch,
         typename KernelParams, typename FusedParams, typename EpilogueParams>
__launch_bounds__(NumThreads)
__global__ void cudaKernel(KernelParams params,
                           FusedParams fusedParams,
                           DistributedParams distParams,
                           EpilogueParams epilogueParams) {
#ifdef __CUDA_ARCH__
  if (__CUDA_ARCH__ != SMArch) return;
#endif
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
  static_assert(kTileK % MaxP == 0,
                "TileK is not a multiple of MaxP");
  static_assert((kTileK/MaxP)%RegK == 0,
                "RegK not a multiple of MaxCols/MaxP");

  constexpr bool kFactorShapeSame  = KernelOptimizations::IsFactorShapeSame (OptLevel);
  constexpr bool kXshSlicesSame    = KernelOptimizations::IsXshSlicesSame   (OptLevel);
  constexpr bool kQMultipleOfTileQ = KernelOptimizations::IsQMultipleOfTileQ(OptLevel);
  constexpr bool kPMultipleOfTileP = KernelOptimizations::IsPMultipleOfTileP(OptLevel);
  constexpr bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);
  constexpr bool kMMultipleOfTileM = KernelOptimizations::IsMMultipleOfTileM(OptLevel) || TileM ==1;
  constexpr bool kQLeTileQ         = KernelOptimizations::IsQLeTileQ        (OptLevel);
  constexpr bool kTileKSame        = KernelOptimizations::IsTileKSame       (OptLevel);

  //Vector Load types based on alignments 
  using XVecT = typename std::conditional<!kKMultipleOfTileK || XAlignment == 1, ElemT, 
                typename std::conditional<XAlignment == 2, Vec2T, 
                                          Vec4T>::type>::type;

  const bool LoadFullFactor = kPMultipleOfTileP && kKMultipleOfTileK &&
                              MaxP > 2 && MaxQ > 2 && TileP >= MaxP && 
                              TileQ >= MaxQ && (MaxP*MaxQ) % 4 == 0;
  using FVecT = typename std::conditional<LoadFullFactor , Vec4T,
                typename std::conditional<FAlignment == 1, ElemT,
                typename std::conditional<FAlignment == 2, Vec2T,
                                          Vec4T>::type>::type>::type;

  GetBatchedData<KernelBatch, ElemT, KernelParams, EpilogueParams> batchedData;

  const uint32_t batch = blockIdx.z;

  const Matrix X = batchedData.getXBatch(params, batch);
  const Matrix Y = batchedData.getYBatch(params, batch);

  static_assert(!(kQLeTileQ && kQMultipleOfTileQ),
                "Both QLeTileQ and QMultipleOfTileQ cannot be true at same time");

  const uint Q = (kFactorShapeSame) ? MaxQ : params.problem.f(0).q();
  const uint P = (kFactorShapeSame) ? MaxP : params.problem.f(0).p();

  const uint XshSlices = getXshSlices<OptLevel, kTileK, MaxP>(params);
  const uint XSlices   = getXSlices  <OptLevel, MaxQ>(Y, params);

  const uint QThreads  = getQThreads <kXshSlicesSame, RegK>(XshSlices);
  const uint QByTileQ  = getQByTileQ <kQLeTileQ, TileQ>(Q);
  const uint TileK     = getXTileK   <OptLevel, kTileK>(params);
  // const uint ShTileK   = XshSlices*TileP;

  const fastKronOp OpX = (kmmType == FastKronMMType::MKM) ? kOpX : swapFastKronOp<kOpX>();
  const fastKronOp OpF = (kmmType == FastKronMMType::MKM) ? kOpF : swapFastKronOp<kOpF>();
  const fastKronOp OpY = (kmmType == FastKronMMType::MKM) ? fastKronOp_N : fastKronOp_T;

  const uint bid_x = (OpX == fastKronOp_N) ? blockIdx.x : ((KernelBatch == KernelBatchType::Normal ? blockIdx.z * 32768 : 0) + 
                                                            blockIdx.y);
  const uint bid_y = (OpX == fastKronOp_N) ? blockIdx.y : blockIdx.x;
  const uint tid   = threadIdx.x;

  //TODO: Make this Coord2D
  const uint tileQ = getTileQ(bid_x, QByTileQ);
  const uint tileK = getTileK(bid_x, QByTileQ);

  const YElem yElem = getYElem(tid, OpY, NumThreads, QThreads, MaxP, TileM, kTileK, TileQ, RegM, RegK, RegQ);
  const uint tileM = bid_y * TileM;
  //TODO: is this condition optimized for OptLevel == 3?
  if (tileM >= X.m() || tileK * TileK >= X.n()) return;
  
  Slice<ElemT, OpX> XTile(tileM, tileK * TileK,
                          (kMMultipleOfTileM || TileM == 1) ? TileM : MIN(TileM, X.m() - tileM), 
                          (kKMultipleOfTileK) ? TileK : MIN(X.n()-tileK * TileK, TileK),
                          P, TileP,
                          X);

  extern __shared__ ElemT sharedStorage[];//[TileM*ShTileK + TileP*TileQ];
  
  //If X or F are Op_T then transpose then in shared memory
  using XShared = ShiftShared<OpY, ElemT, kXshSlicesSame, 
                              TileM, kTileK/MaxP, TileP>;
  using FShared = DirectShared<OpY, ElemT, TileP, TileQ>;
  XShared Xsh(&sharedStorage[0], kTileK/MaxP * TileP);
  FShared Fsh(&sharedStorage[Xsh.numel()]);

  register YRegisters<fastKronOp_N, ElemT, RegM, RegK, RegQ> yReg; //Layout is not used in CUDA

  for (uint32_t tileP = 0; tileP < P; tileP += TileP) {
    //Loop iterates only once when FusedFacs == 1
    //Load X to shared memory
    shiftXgToXsh<kMMultipleOfTileM, kXshSlicesSame, kPMultipleOfTileP, TileP, ElemT, XVecT, OpX>
                (NumThreads, RegK, tileP, tid, XTile, Xsh);
    #pragma unroll
    for (int _fac = FusedFacs - 1; _fac >= 0; _fac--) {
      int fac = 0;
      if (OpY == fastKronOp_N)      fac = _fac;
      else if (OpY == fastKronOp_T) fac = FusedFacs - 1 - _fac;
      
      const Factor F(P, Q, batchedData.getFBatch(params, fac, batch).data());
      //Load F to shared memory
      directFgToFsh<kPMultipleOfTileP, kQMultipleOfTileQ, ElemT, FVecT, OpF>
                    (NumThreads, tid, tileP, tileQ, F, Fsh);

      __syncthreads();

      //Zero out register results for fusion iterations
      if (FusedFacs > 1) yReg.zero();
      if (kFactorShapeSame ||
          ((kKMultipleOfTileK || yElem.k() < MIN(XshSlices, XSlices - tileK * XshSlices)) &&
           (kQMultipleOfTileQ || yElem.q() < MIN(TileQ, Q - tileQ * TileQ)) &&
           (kMMultipleOfTileM || yElem.m() < XTile.m()))
          ) {
        /*register*/ XRegisters<fastKronOp_N, ElemT, TileM, RegK, TileP> Xr; //Layout is not used in CUDA
        /*register*/ FRegisters<ElemT, TileP, RegQ> Fr;

        mainMMA(XTile.m(), Xsh, Fsh, yReg, Xr, Fr, yElem);
      }

      if (FusedFacs > 1 && _fac > 0) {
        __syncthreads();
        //Store C to shared memory using shift method
        fusionYrToXSh(XTile.m(), F, Fsh, Xsh, yReg, yElem);
      }

      __syncthreads();
  }}

  constexpr uint32_t StLen = storeVectorLen<OpY, kMMultipleOfTileM, kKMultipleOfTileK, 
                                              FusedFacs, XAlignment, RegM, RegK>();

  if (OpY == fastKronOp_N) {
    #pragma unroll
    for (uint rm = 0; rm < RegM; rm++) {
    #pragma unroll
    for (uint tq = 0; tq < RegQ; tq++) {
    #pragma unroll
    for (uint tk = 0; tk < RegK; tk += StLen) {
      const uint glM = rm + yElem.m() + tileM;
      if (kMMultipleOfTileM || (rm + yElem.m() < XTile.m())) {
      if ((!kKMultipleOfTileK && yElem.k() + tk >= MIN(XshSlices, XSlices - tileK * XshSlices)) || 
          (!kQMultipleOfTileQ && yElem.q() + tq >= MIN(TileQ, Q - tileQ * TileQ))) continue;

      uint glK;
      ElemT* yPtr;
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
          glK += tileQ * XSlices * TileQ;
      }}

    
      if (DistributeToGPUs) {
        yPtr = p2pStoreAddress<ElemT, DistributedParams>(distParams, Y, glM, glK);
      } else {
        cIdx = glM * Y.n() + glK;
        yPtr = Y.data<ElemT>(glM, glK, OpY);
        if (params.kp_idx == FusedFacs - 1) {
          #pragma unroll
          for (int i = 0; i < StLen; i++) {
            ElemT yelem = 0;
            if (OpY == fastKronOp_N) {
              yelem = yReg.at(rm, tk+i, tq);
            } else if (OpY == fastKronOp_T) {
              yelem = yReg.at(rm+i, tk, tq);
            }
            yelem = epilogue(epilogueParams, batchedData, Y, batch, cIdx+i, yelem);
            if (OpY == fastKronOp_N) {
              yReg.set(rm, tk + i, tq, yelem);
            } else if (OpY == fastKronOp_T) {
              yReg.set(rm+i, tk, tq, yelem);
            }
          }
        }
      }
      stVecYReg<OpY, StLen>(yPtr, yReg, rm, tk, tq);
    }}}}
  } else if (OpY == fastKronOp_T) {
    #pragma unroll
    for (uint tq = 0; tq < RegQ; tq++) {
    #pragma unroll
    for (uint tk = 0; tk < RegK; tk++) {
    #pragma unroll
    for (uint rm = 0; rm < RegM; rm+=StLen) {
      const uint glM = rm + yElem.m() + tileM;
      if (kMMultipleOfTileM || (rm + yElem.m() < XTile.m())) {
      if ((!kKMultipleOfTileK && yElem.k() + tk >= MIN(XshSlices, XSlices - tileK * XshSlices)) || 
          (!kQMultipleOfTileQ && yElem.q() + tq >= MIN(TileQ, Q - tileQ * TileQ))) continue;

      uint glK;
      ElemT* yPtr;
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
          glK += tileQ * XSlices * TileQ;
      }}

    
      if (DistributeToGPUs) {
        yPtr = p2pStoreAddress<ElemT, DistributedParams>(distParams, Y, glM, glK);
      } else {
        cIdx = glK * Y.m() + glM;
        yPtr = Y.data<ElemT>(glM, glK, OpY);
        if (epilogueParams.isLastFactor) {
          //TODO: Combine this same in the above OpY case
          #pragma unroll
          for (int i = 0; i < StLen; i++) {
            ElemT yelem = 0;
            if (OpY == fastKronOp_N) {
              yelem = yReg.at(rm, tk+i, tq);
            } else if (OpY == fastKronOp_T) {
              yelem = yReg.at(rm+i, tk, tq);
            }
            ElemT yElemp = yelem;
            yelem = epilogue(epilogueParams, batchedData, Y, batch, cIdx+i, yelem);
            if (OpY == fastKronOp_N) {
              yReg.set(rm, tk + i, tq, yelem);
            } else if (OpY == fastKronOp_T) {
              yReg.set(rm+i, tk, tq, yelem);
            }
          }
        }
      }
      stVecYReg<OpY, StLen>(yPtr, yReg, rm, tk, tq);
    }}}}
  }
}