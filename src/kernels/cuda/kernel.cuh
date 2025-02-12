#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/amd_detail/host_defines.h>
#endif

#include "config.h"
#include "kernels/hw_details.h"
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

template<FMAInstType FMAInst>
CUDA_DEVICE
YElem getYElem(const uint32_t tid, fastKronOp OpY, const uint32_t NumThreads, uint32_t QThreads,
               const uint32_t MaxP,
               const uint32_t TileM, const uint32_t kTileK, const uint32_t TileQ,
               const uint32_t RegM,  const uint32_t RegK,   const uint32_t RegQ) {
  if (OpY == fastKronOp_N) {
    const uint MThreads  = (TileM == 1) ? NumThreads : (TileQ/RegQ) * ((kTileK/MaxP)/RegK);
    const uint yM   = (MThreads >= NumThreads) ? 0 : ((tid / MThreads) * RegM);
    const uint wid = tid/CUDA_WARP_SIZE;
    const uint QWarps = 2;//QThreads/CUDA_WARP_SIZE;

    if (FMAInst == FMAInstType::Tensor884) {
      
      const uint yQ   = (wid / QWarps) * RegQ; //wid / 2 * 8 = 0,0,0,0 ... ,
      const uint yK   = (wid % QWarps) * RegK; //wid % 2 * 8 = 0,0,..,8,8,...,
      return YElem(yM, yQ, yK);

    } else {
      const uint wQ   = (wid / QWarps) * 8;
      const uint wK   = (wid % QWarps) * 8;
      const uint yQ   = ((tid % CUDA_WARP_SIZE) / 4) * RegQ; //tid / 4 * 1 = 0,0,0,..,1,1,1,2,2,2
      const uint yK   = ((tid % CUDA_WARP_SIZE) % 4) * RegK; //(tid % 4) * 2 = 0,2,4,..8,
    
      return YElem(yM, wQ + yQ, wK + yK);
    }
  } else if (OpY == fastKronOp_T) {
    QThreads = QThreads * (TileM/RegM);
    const uint KThreads  = TileM/RegM;
    const uint yQ   = (tid / QThreads) * RegQ;
    const uint yM   = ((tid % QThreads) % KThreads) * RegM;
    const uint yK   = ((tid % QThreads) / KThreads) * RegK;

    return YElem(yM,yQ,yK);
  }
  return YElem(0,0,0);
}

template<uint SMArch, FMAInstType FMAInst, typename ElemT, typename Vec2T, typename Vec4T,
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

  GetBatchedData<KernelBatch, ElemT, KernelParams, FusedParams, EpilogueParams> batchedData;

  const uint32_t batch = (KernelBatch == KernelBatchType::Normal) ? 0 : (params.startGridz + blockIdx.z);

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

  const uint bid_x = (OpX == fastKronOp_N) ? params.startGridx + blockIdx.x: 
                                             params.startGridy + blockIdx.y;
  const uint bid_y = (OpX == fastKronOp_N) ? params.startGridy + blockIdx.y:
                                             params.startGridx + blockIdx.x;
  const uint tid   = threadIdx.x;

  //TODO: Make this Coord2D
  const uint tileQ = getTileQ(bid_x, QByTileQ);
  const uint tileK = getTileK(bid_x, QByTileQ);

  const YElem yElem = getYElem<FMAInst>(tid, OpY, NumThreads, QThreads, MaxP, TileM, kTileK, TileQ, RegM, RegK, RegQ);
  const uint tileM = bid_y * TileM;

  if ((tileM >= X.m()) || 
      (tileK * TileK >= X.n()))
      return;
  
  Slice<ElemT, OpX> XTile(tileM, tileK * TileK,
                          (kMMultipleOfTileM || TileM == 1) ? TileM : MIN(TileM, X.m() - tileM), 
                          (kKMultipleOfTileK) ? TileK : MIN(X.n()-tileK * TileK, TileK),
                          P, TileP,
                          X);

  extern __shared__ ElemT sharedStorage[];//[TileM*ShTileK + TileP*(TileQ+Padding)];
  //Padding is only applied for KMM
  
  //If X or F are Op_T then transpose then in shared memory
  using XShared = ShiftShared<OpY, ElemT, kXshSlicesSame, 
                              TileM, kTileK/MaxP, TileP>;
  using FShared = DirectShared<OpY, ElemT, TileP, TileQ>;
  XShared Xsh(&sharedStorage[0], kTileK/MaxP * TileP);
  FShared Fsh(&sharedStorage[Xsh.numel()]);

  constexpr uint32_t CoreK = (FMAInst == FMAInstType::Tensor884) ? 8 : 1;
  constexpr uint32_t CoreQ = (FMAInst == FMAInstType::Tensor884) ? 8 : 1;
  constexpr uint32_t CoreP = (FMAInst == FMAInstType::Tensor884) ? 4 : TileP;

  register YRegisters<fastKronOp_N, ElemT, RegM, 2, 1> yReg; //RegK/CoreK, RegQ/CoreQ> yReg; //Layout is not used in CUDA

  for (uint32_t tileP = 0; tileP < P; tileP += TileP) {
    //Loop iterates only once when FusedFacs == 1
    //Load X to shared memory
    shiftXgToXsh<kMMultipleOfTileM, kXshSlicesSame, kPMultipleOfTileP, TileP, ElemT, XVecT, OpX>
                (NumThreads, RegK, tileP, tid, XTile, Xsh);
    #pragma unroll
    for (int fac = ((FusedFacs == 1) ? 1 : params.problem.n()) - 1; fac >= 0; fac--) {
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
        for (int coreP = 0; coreP < TileP; coreP += CoreP) {
          /*register*/ XRegisters<fastKronOp_N, ElemT, TileM, RegK/CoreK, 1> Xr; //Layout is not used in CUDA
          /*register*/ FRegisters<ElemT, 1, RegQ/CoreQ> Fr;

          mainMMA<FMAInst>(XTile.m(), Xsh, Fsh, yReg, Xr, Fr, yElem, coreP);
          // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          //   printf("216 %d: %lf %lf %lf %lf;\n", params.kp_idx, Xsh.at(0,0), Xsh.at(0,1), Xsh.at(0,2), yReg.data[0]);
        }
      }

      if (FusedFacs > 1 && fac > 0) {
        __syncthreads();
        //Store C to shared memory using shift method
        fusionYrToXSh(XTile.m(), F, Fsh, Xsh, yReg, yElem);
        
        Matrix fusedInter = batchedData.getIntermediateBatch(fusedParams, fac, batch);
        if (fusedInter.data() != nullptr) {
          storeY<OpY, RegM, RegK, RegQ, TileQ,
                kMMultipleOfTileM, kKMultipleOfTileK, kQMultipleOfTileQ, FusedFacs, DistributeToGPUs, XAlignment,
                ElemT> (fac, batch, XshSlices, XSlices, tileM, tileK, tileQ, P, Q, XTile, Xsh, fusedInter, yElem, yReg,
                        params, fusedParams, distParams, epilogueParams, batchedData);
        }
      }

      __syncthreads();
  }}

  __shared__ double transpose[64 * 2];

  double* warpTr = &transpose[(tid/CUDA_WARP_SIZE)*2*CUDA_WARP_SIZE];
  uint32_t lane = (tid % CUDA_WARP_SIZE);
  ((double2*)warpTr)[lane] = double2{yReg.data[0], yReg.data[1]};

  __syncthreads();

  YElem yElem2 = getYElem<FMAInstType::SIMT>(tid, OpY, NumThreads, XshSlices/2, MaxP, TileM, kTileK, TileQ, RegM, 2, 1);

  yReg.data[0] = warpTr[lane/4 + (lane%4) * 16];
  yReg.data[1] = warpTr[lane/4 + (lane%4) * 16 + 8];

  storeY<OpY, RegM, 2, 1, TileQ,
         kMMultipleOfTileM, kKMultipleOfTileK, kQMultipleOfTileQ, FusedFacs, DistributeToGPUs, XAlignment,
         ElemT> (0, batch, XshSlices, XSlices, tileM, tileK, tileQ, P, Q, XTile, Xsh, Y, yElem2, yReg,
                 params, fusedParams, distParams, epilogueParams, batchedData);
}