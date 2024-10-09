#include <iostream>
#include <numeric>
#include <cmath>

#include "kmm/kmmalgo.h"
#include "config.h"
#include "utils/utils.h"
#include "kernels/kernel_opt.h"

enum KernelMode {
  KernelModeTuning,
  KernelModeNormal,
};

namespace KernelBatchType {
  enum Ty {
    Normal,
    StridedBatched,
    Batch
  };
}

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
  float  get(float) const {return f;}
  CUDA_DEVICE
  long   get(long) const {return l;}
  CUDA_DEVICE
  int    get(int) const {return i;}
  CUDA_DEVICE
  double get(double) const {return d;}
};

struct EpilogueParams {
  AllTypes alpha;
  AllTypes beta;
  const void * __restrict__ glD;
  
  EpilogueParams(): alpha(1.0f), beta(0.0f), glD(nullptr) {}

  EpilogueParams(AllTypes alpha, AllTypes beta, const void* glD) :
    alpha(alpha), beta(beta), glD(glD) {}

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

  //TODO: Change to alphaAs<type>
  template<typename ElemT>
  CUDA_DEVICE
  ElemT        getAlpha() const {return alpha.get((ElemT)0);}
  
  template<typename ElemT>
  CUDA_DEVICE
  ElemT        getBeta()  const {return beta.get((ElemT)0);}
  
  template<typename ElemT>
  CUDA_DEVICE
  const ElemT* getD()     const {return (const ElemT*)glD;}
  
  template<typename ElemT>
  CUDA_DEVICE
  const ElemT* z()     const {return (const ElemT*)glD;}
};

struct EpilogueStridedBatchedParams : public EpilogueParams {
  uint64_t strideZ;

  EpilogueStridedBatchedParams() : EpilogueParams(), strideZ(0) {}
  EpilogueStridedBatchedParams(AllTypes alpha, AllTypes beta, const void* glD, uint32_t strideZ) : 
    EpilogueParams(alpha, beta, glD), strideZ(strideZ) {}

  template<typename ElemT>
  static EpilogueStridedBatchedParams create() {
    return EpilogueStridedBatchedParams(AllTypes((ElemT)1.0f), 
                          AllTypes((ElemT)0.0f),
                          nullptr, 0);
  }

  template<typename ElemT>
  static EpilogueStridedBatchedParams create(const ElemT alpha, const ElemT beta,
                               const ElemT* glD, uint64_t strideZ) {
    return EpilogueStridedBatchedParams(AllTypes(alpha), AllTypes(beta),
                                        (const void*)glD, strideZ);
  }

  CUDA_DEVICE
  uint64_t getStrideZ() {return strideZ;}
};

struct CPUCaches {
  void** TileXs, **TileYs, **TileFs;

  CPUCaches(void** TileXs, void** TileFs, void** TileYs) : 
          TileXs(TileXs), TileYs(TileYs), TileFs(TileFs) {}
};

template<typename KMMProblemT>
struct KernelParams {
  KMMProblemT problem;
  
  Matrix tileX;
  Factor tileF;
  
  uint32_t XshSlices;
  uint32_t XSlices;
  
  const uint kp_idx;
  KernelMode execMode;
  CPUCaches* caches;

  KernelParams(KMMProblemT problem_, CPUCaches* caches,
               Matrix tileX, Factor tileF, uint kp_idx, KernelMode execMode) :
               problem(problem_), 
               tileX(tileX), tileF(tileF),
               XshSlices(tileX.n()/problem_.f(0).p()),
               XSlices(problem_.x().n()/problem_.f(0).p()),
               kp_idx(kp_idx), execMode(execMode), caches(caches) {}
};

template<typename KMMProblemT>
struct FusedParams {
  uint XShFusedSlices;
  uint XglFusedSlices;
  static const uint32_t NumFused = KMMProblemT::MaxFactors;

  FusedParams(KMMProblemT problem, const uint TileSizeColsA) {
    const Factor factorPower = std::reduce(problem.fs(), problem.fs() + problem.n(), Factor(1,1), [](Factor prev, Factor curr) {
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