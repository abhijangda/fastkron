#include <iostream>
#include <numeric>
#include <cmath>

#include "kmm/kmmalgo.h"
#include "config.h"

enum KernelMode {
  KernelModeTuning,
  KernelModeNormal,
};

struct KernelOptimizations {
  enum Optimization {
    None = 0,
    XshSlicesSame    = 1 << 0,
    QMultipleOfTileQ = 1 << 1,
    PMultipleOfTileP = 1 << 2,
    KMultipleOfTileK = 1 << 3,
    QLeTileQ         = 1 << 4,
    TileKSame        = 1 << 5,
    FactorShapeSame  = 1 << 6,
    NumOptimizations = 1 << 7
  };

  CUDA_DEVICE_HOST
  static constexpr uint OptLevel0() {
    return Optimization::None;
  }

  CUDA_DEVICE_HOST
  static constexpr uint OptLevel1() {
    return OptLevel0()                 |
           Optimization::XshSlicesSame
           ;
  }

  CUDA_DEVICE_HOST
  static constexpr uint OptLevel2() {
    return OptLevel1()                    | 
           Optimization::KMultipleOfTileK |
           Optimization::QMultipleOfTileQ |
           Optimization::PMultipleOfTileP
           ;
  }

  CUDA_DEVICE_HOST
  static constexpr uint OptLevel3() {
    return OptLevel2()                   |
           Optimization::FactorShapeSame |
           Optimization::TileKSame
           ;
  }

  CUDA_DEVICE_HOST
  static constexpr uint MaxOptLevel() {
    return 3;
  }

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

#pragma once

union AllTypes {
  int i;
  long l;
  float f;
  double d;

  CUDA_DEVICE_HOST
  AllTypes()                {}
  AllTypes(float  f) : f(f) {}
  AllTypes(long   l) : l(l) {}
  AllTypes(int    i) : i(i) {}
  AllTypes(double d) : d(d) {}

  CUDA_DEVICE
  float  get(float  p) const {return f;}
  CUDA_DEVICE
  long   get(long   p) const {return l;}
  CUDA_DEVICE
  int    get(int    p) const {return i;}
  CUDA_DEVICE
  double get(double p) const {return d;}
};

struct EpilogueParams {
  const void * __restrict__ glD;
  AllTypes alpha;
  AllTypes beta;
  
  EpilogueParams(): alpha(1.0f), beta(0.0f), glD(nullptr) {}

  EpilogueParams(AllTypes alpha, AllTypes beta, const void* glD) :
    glD(glD), alpha(alpha), beta(beta) {}

  template<typename ElemT>
  static EpilogueParams create() {
    return EpilogueParams(AllTypes((ElemT)1.0f), 
                          AllTypes((ElemT)0.0f),
                          nullptr);
  }

  template<typename ElemT>
  static EpilogueParams create(const ElemT alpha, 
                               const ElemT beta,
                               const ElemT* glD) {
    return EpilogueParams(AllTypes(alpha),
                          AllTypes(beta),
                          (const void*)glD);
  }

  template<typename ElemT>
  CUDA_DEVICE
  ElemT        getAlpha() const {return alpha.get((ElemT)0);}
  
  template<typename ElemT>
  CUDA_DEVICE
  ElemT        getBeta()  const {return beta.get((ElemT)0);}
  
  template<typename ElemT>
  CUDA_DEVICE
  const ElemT* getD()     const {return (const ElemT*)glD;}
};

template<uint Fused>
struct KernelParams {
  KMMProblemT<Fused> problem;
  const uint kp_idx;
  KernelMode execMode;
  
  Matrix tileX;
  Factor tileF;
  
  uint32_t XshSlices;
  uint32_t XSlices;

  KernelParams(KMMProblem problem_, Matrix tileX, Factor tileF, uint kp_idx, KernelMode execMode) :
               problem(problem_), tileX(tileX), tileF(tileF),
               XshSlices(tileX.n()/problem_.f(0).p()),
               XSlices(problem_.x().n()/problem_.f(0).p()),
               kp_idx(kp_idx), execMode(execMode) {}
};

template<uint Fused>
struct FusedParams {
  uint KronColsPower;
  uint XShFusedSlices;
  uint XglFusedSlices;
  
  FusedParams(KMMProblem problem, const uint TileSizeColsA) {
    const Factor factorPower = std::reduce(problem.fs(), problem.fs() + Fused, Factor(1,1), [](Factor prev, Factor curr) {
      return Factor(prev.p() * curr.p(), prev.q() * curr.q());
    });

    XShFusedSlices = TileSizeColsA/factorPower.p();
    XglFusedSlices = problem.k()/factorPower.p();
  }
};

struct DistributedParams {
  //TODO: Set gpuResults for 16 GPUs
  void* gpuResults0;
  void* gpuResults1; 
  void* gpuResults2;
  void* gpuResults3;
  void* gpuResults4;
  void* gpuResults5; 
  void* gpuResults6;
  void* gpuResults7;
 
  const uint gr, gc;
  const uint gpusInK;
  const uint ColsA;
  const uint ColsC;
  const uint LocalKrons;

  uint UVAColsRatioKronRowsSquare;
  uint perGPUKByNumGPUs;
  uint perGPUKByKronRows;
  uint perGPUNByNumGPUs;
  uint perGPUNByKronRows;
  uint perGPUNByKronCols;
  uint ColsAByKronRows;
  uint ColsCByKronCols;
  uint gcMulUVAColsRatioKronRowsSquare;
  uint ColsCByKronRowsPower;
  uint ColsCByKronColsPower;

  DistributedParams() : gr(0), gc(0), gpusInK(1), ColsA(0), ColsC(0), LocalKrons(1) {} 
  
  DistributedParams(const uint gr_, const uint gc_, const uint gpusInK_,   
                    const uint ColsA_, const uint ColsC_, 
                    const uint PerGPUK_, const uint PerGPUN_, 
                    const Factor* Factors, const uint LocalKrons_) :
    gr(gr_), gc(gc_), gpusInK(gpusInK_), ColsA(ColsA_), ColsC(ColsC_),
    LocalKrons(LocalKrons_) {
    
    const Factor factorPower = std::reduce(Factors, Factors + LocalKrons_, Factor(1,1), [](Factor prev, Factor curr) {
      return Factor(prev.p() * curr.p(), prev.q() * curr.q());
    });
    const uint KronColsPower = factorPower.q();
    const uint KronRowsPower = factorPower.p();

    UVAColsRatioKronRowsSquare = PerGPUN_/KronColsPower;
    perGPUKByNumGPUs = PerGPUK_/gpusInK_;
    perGPUNByNumGPUs = PerGPUN_/gpusInK_;
    perGPUNByKronCols = PerGPUN_/Factors[LocalKrons_-1].q(); //Same as perGPUKByKronCols
    gcMulUVAColsRatioKronRowsSquare = gc*UVAColsRatioKronRowsSquare;
    ColsCByKronCols = ColsC_/Factors[LocalKrons_-1].q();
    ColsCByKronColsPower = ColsC_/KronColsPower;
    perGPUNByKronRows = PerGPUN_/Factors[LocalKrons_-1].p();
    perGPUKByKronRows = PerGPUK_/Factors[LocalKrons_-1].p();
    ColsCByKronRowsPower = ColsC_/KronRowsPower;
    ColsAByKronRows = ColsA_/Factors[LocalKrons_-1].p();
  }

  void updateGPUResults(void** gpuResults_) {
    setGPUResults(0, gpuResults0, gpuResults_);
    setGPUResults(1, gpuResults1, gpuResults_);
    setGPUResults(2, gpuResults2, gpuResults_);
    setGPUResults(3, gpuResults3, gpuResults_);
    setGPUResults(4, gpuResults4, gpuResults_);
    setGPUResults(5, gpuResults5, gpuResults_);
    setGPUResults(6, gpuResults6, gpuResults_);
    setGPUResults(7, gpuResults7, gpuResults_);
  }

  void setGPUResults(uint idx, void*& thisResults, void** gpuResults) {
    if (idx < gpusInK)
      thisResults = gpuResults[idx];
    else
      thisResults = nullptr;
  }

  CUDA_DEVICE void* getLocalGPUResult(uint gc) const {
    switch(gc) {
      case 0: return gpuResults0;
      case 1: return gpuResults1;
      case 2: return gpuResults2;
      case 3: return gpuResults3;
      case 4: return gpuResults4;
      case 5: return gpuResults5;
      case 6: return gpuResults6;
      case 7: return gpuResults7;
      default: return nullptr;
      //TODO: for all 16 GPUs
    }
  }

  uint32_t proc() const {
    return gr * gpusInK + gc;
  }
};