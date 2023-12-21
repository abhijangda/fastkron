#include <iostream>
#include <numeric>
#include <cmath>

#include "kmm/kmmalgo.h"

#pragma once

union AllTypes {
  int i;
  long l;
  float f;
  double d;

  __host__ __device__
  AllTypes()                {}
  AllTypes(float f)  : f(f) {}
  AllTypes(long l)   : l(l) {}
  AllTypes(int i)    : i(i) {}
  AllTypes(double d) : d(d) {}

  __device__
  float  get(float  p) {return f;}
  __device__
  long   get(long   p) {return l;}
  __device__
  int    get(int    p) {return i;}
  __device__
  double get(double p) {return d;}
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
  __device__
  ElemT        getAlpha() {return alpha.get((ElemT)0);}
  
  template<typename ElemT>
  __device__
  ElemT        getBeta()  {return beta.get((ElemT)0);}
  
  template<typename ElemT>
  __device__
  const ElemT* getD()     {return (const ElemT*)glD;}
};

template<uint Fused>
struct KernelParams {
  const uint m;
  const uint l;
  const uint k;
  uint ps[Fused];
  uint qs[Fused];
  const void * __restrict__ x;
  const void * __restrict__ fs[Fused];
  void       * __restrict__ y;
  const uint kp_idx;

  KernelParams(KMMProblem problem, uint kp_idx) :
               m(problem.m()), l(problem.l()), k(problem.k()),
               x(problem.x.ptr()), y(problem.y.ptr()), kp_idx(kp_idx) {
    for (int i = 0; i < Fused; i++) {
      ps[Fused - 1 - i] = problem.fs[i].p();
      qs[Fused - 1 - i] = problem.fs[i].q();
      fs[Fused - 1 - i] = problem.fs[i].ptr();
    }
  }
};

template<uint Fused>
struct FusedParams {
  uint KronColsPower;
  uint UVAColsRatioKronColsSquare;
  uint ColsCByKronColsPower;
  
  FusedParams(KMMProblem problem, const uint TileSizeColsA) {
    KronColsPower = (uint)std::pow((double)problem.fs[0].q(), (double)Fused);
    UVAColsRatioKronColsSquare = TileSizeColsA/KronColsPower;
    ColsCByKronColsPower = problem.l()/KronColsPower;
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
    perGPUNByKronCols = PerGPUN_/Factors[LocalKrons_-1].q();
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

  __device__ __forceinline__ void* getLocalGPUResult(uint gc) {
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
};