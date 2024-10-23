#pragma once

/**
 * A KMMKernel can be compiled with one or more of the following.
 * Each optimization optimizes the kernel for a case of problem shapes.
 * There are 4 optimization levels 0 to 3. Higher opt level has more 
 * optimizations and each level add extra optimizations over the prev level.
 */
struct KernelOptimizations {
  /**
   * An enum of each Optimization 
   */
  enum Optimization {
    //No optimization, i.e. a general kernel.
    None = 0,
    //No. of slices of tile of X cols is same as the slices of kernel's tileK.
    XshSlicesSame    = 1 << 0,
    //The problem Q is a multiple of kernel's TileF.q()
    QMultipleOfTileQ = 1 << 1,
    //The problem P is a multiple of kernel's TileF.p()
    PMultipleOfTileP = 1 << 2,
    //The problem's X.cols is a multiple of kernel's TileX.n()
    KMultipleOfTileK = 1 << 3,
    MMultipleOfTileM = 1 << 4,
    //The problem Q is less than kernel's TileF.q()
    QLeTileQ         = 1 << 5,
    //Kernel is invoked with same TileK as the template TileK
    TileKSame        = 1 << 6,
    //Problem's factor has same shape as kernel's MaxF
    FactorShapeSame  = 1 << 7,
    //Number of Optimizations
    NumOptimizations = 1 << 8
  };

  /**
   * OptLevel0() - Return a bitwise OR of optimizations at level 0.
   *               At level 0, there are no optimization and a kernel
   *               will run for any X and F shape.
   */
  CUDA_DEVICE_HOST
  static constexpr uint OptLevel0() {
    return Optimization::None;
  }

  /**
   * OptLevel1() - Return a bitwise OR of optimizations at level 1.
   *               At level 1, slices of tile of X cols are same as kernel's tile slices.
   */
  CUDA_DEVICE_HOST
  static constexpr uint OptLevel1() {
    return OptLevel0()                 |
           Optimization::XshSlicesSame
           ;
  }

  /**
   * OptLevel2() - Return a bitwise OR of optimizations at level 2.
   *               At level 2, problem's K, Q, and P must be multiple of kernel's 
   *               TileK, TileQ and TileP respectively.
   */
  CUDA_DEVICE_HOST
  static constexpr uint OptLevel2() {
    return OptLevel1()                    | 
           Optimization::KMultipleOfTileK |
           Optimization::QMultipleOfTileQ |
           Optimization::PMultipleOfTileP
           ;
  }

  /**
   * OptLevel3() - Return a bitwise OR of optimizations at level 3.
   *               At level 3, problem's factor has same shape as kernel's and 
   *               kernel is invoked with same TileK as kernel's template.
   */
  CUDA_DEVICE_HOST
  static constexpr uint OptLevel3() {
    return OptLevel2()                   |
           Optimization::FactorShapeSame |
           Optimization::TileKSame       |
           Optimization::MMultipleOfTileM
           ;
  }

  /**
   * MaxOptLevel() - Return maximum optimization level, i.e. 3.
   */
  CUDA_DEVICE_HOST
  static constexpr uint MaxOptLevel() {
    return 3;
  }

  /**
   * getOptimizations() - Return bitwise OR of optimizations at given level.
   * @optLevel: Optimization level.
   */
  CUDA_DEVICE_HOST
  static constexpr uint getOptimizations(uint optLevel) {
    switch(optLevel) {
      case 0: return OptLevel0();
      case 1: return OptLevel1();
      case 2: return OptLevel2();
      case 3: return OptLevel3();
      default:
        return 0;
    }
  }

  /**
   * isEnabled() - Return true if an optimization is enabled in an optimization level.
   * @optLevel: Optimization level.
   * @specl: Optimization.
   */
  CUDA_DEVICE_HOST
  static constexpr bool isEnabled(uint optLevel, Optimization specl) {
    return (getOptimizations(optLevel) & specl) == specl;
  }

  CUDA_DEVICE_HOST
  static constexpr bool IsXshSlicesSame(uint optLevel) {
    return isEnabled(optLevel, Optimization::XshSlicesSame);
  }

  CUDA_DEVICE_HOST
  static constexpr bool IsQMultipleOfTileQ(uint optLevel) {
    return isEnabled(optLevel, Optimization::QMultipleOfTileQ);
  }

  CUDA_DEVICE_HOST
  static constexpr bool IsPMultipleOfTileP(uint optLevel) {
    return isEnabled(optLevel, Optimization::PMultipleOfTileP);
  }

  CUDA_DEVICE_HOST
  static constexpr bool IsKMultipleOfTileK(uint optLevel) {
    return isEnabled(optLevel, Optimization::KMultipleOfTileK);
  }

  CUDA_DEVICE_HOST
  static constexpr bool IsMMultipleOfTileM(uint optLevel) {
    return isEnabled(optLevel, Optimization::MMultipleOfTileM);
  }

  CUDA_DEVICE_HOST
  static constexpr bool IsQLeTileQ        (uint optLevel) {
    return isEnabled(optLevel, Optimization::QLeTileQ);
  }

  CUDA_DEVICE_HOST
  static constexpr bool IsTileKSame       (uint optLevel) {
    return isEnabled(optLevel, Optimization::TileKSame);
  }

  CUDA_DEVICE_HOST
  static constexpr bool IsFactorShapeSame (uint optLevel) {
    return isEnabled(optLevel, Optimization::FactorShapeSame);
  }
};

template<uint OptLevel, uint kTileK, uint kP, typename KernelParams>
CUDA_DEVICE_HOST uint32_t getXshSlices(const KernelParams& params) {
constexpr bool kFactorShapeSame = KernelOptimizations::IsFactorShapeSame(OptLevel);
  if (kFactorShapeSame) {
    return kTileK/kP;
  } else {
    return params.XshSlices;
  }
}


template<uint OptLevel, uint kQ, typename KernelParams> 
CUDA_DEVICE_HOST uint32_t getXSlices(const Matrix& Y, const KernelParams& params) {
  //# of slices for a row. Same as X.n()/P but use Y.n()/Q to reduce
  //number of loads as store also requires reading Y.n()
  constexpr bool kFactorShapeSame = KernelOptimizations::IsFactorShapeSame(OptLevel);
  if (kFactorShapeSame) {
    return Y.n()/kQ;
  } else {
    return params.XSlices;
  }
}

template<uint kXshSlicesSame, uint RegK> 
CUDA_DEVICE_HOST uint32_t getQThreads(uint XshSlices) {
  if (kXshSlicesSame) return XshSlices/RegK;
  return DIVUP(XshSlices, RegK);
}

template<uint kQLeTileQ, uint TileQ> 
CUDA_DEVICE_HOST uint32_t getQByTileQ(uint Q) {
  if (kQLeTileQ) {
    return 1;
  }
  return DIVUP(Q, TileQ);
}

template<uint OptLevel, uint kTileK, typename KernelParams> 
CUDA_DEVICE_HOST uint32_t getXTileK(KernelParams& params) {
  constexpr bool kTileKSame = KernelOptimizations::IsTileKSame(OptLevel);
  if (kTileKSame) return kTileK;
  return params.tileX.n();
}