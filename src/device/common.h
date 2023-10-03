#include <iostream>

#ifndef __COMMON_H__
#define __COMMON_H__

#define CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess and e != cudaErrorPeerAccessAlreadyEnabled) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

__host__ __device__ constexpr uint power(const uint x, const uint y) {
  uint result = 1;
  for (uint i = 0; i < y; i++) {
    result = result * x;
  }
  return result;
}

template<typename ElemT, uint NumFusedKerns>
struct KernelParams {
  const uint RowsC;
  const uint ColsC; //TODO: Change to LocalColsC
  const uint ColsA;
  uint KronRows[NumFusedKerns];
  uint KronCols[NumFusedKerns];
  const ElemT * __restrict__ glA;
  const ElemT * __restrict__ glKronMats[NumFusedKerns];
  ElemT       * __restrict__ glC;
  const uint kp_idx;

  KernelParams(const uint RowsC, const uint ColsC, const uint ColsA, const uint KronRows[NumFusedKerns],
               const uint KronCols[NumFusedKerns], const ElemT* glA,
               ElemT* glKronMats[NumFusedKerns], ElemT* glC, uint kp_idx) :
               RowsC(RowsC), ColsC(ColsC), ColsA(ColsA), glA(glA), glC(glC), kp_idx(kp_idx) {
    for (int i = 0; i < NumFusedKerns; i++) {
      this->KronRows[i] = KronRows[i];
      this->KronCols[i] = KronCols[i];
      this->glKronMats[i] = glKronMats[i];
    }
  }
};

template<typename ElemT>
struct DistributedParams {
  //TODO: Set gpuResults for 16 GPUs
  ElemT* gpuResults0;
  ElemT* gpuResults1; 
  ElemT* gpuResults2;
  ElemT* gpuResults3;
  ElemT* gpuResults4;
  ElemT* gpuResults5; 
  ElemT* gpuResults6;
  ElemT* gpuResults7;
 
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

  DistributedParams() : gr(0), gc(0), gpusInK(1), ColsA(0), ColsC(0), LocalKrons(1) {} 
  
  DistributedParams(ElemT** gpuResults_, const uint gr_, const uint gc_, const uint gpusInK_,   
                    const uint ColsA_, const uint ColsC_, 
                    const uint PerGPUK_, const uint PerGPUN_, const uint KronCols_, const uint KronRows_, const uint LocalKrons_) :
    gr(gr_), gc(gc_), gpusInK(gpusInK_), ColsA(ColsA_), ColsC(ColsC_),
    LocalKrons(LocalKrons_) {
    
    const uint KronRowsPower = power(KronRows_, LocalKrons_); //32
    UVAColsRatioKronRowsSquare = PerGPUK_/KronRowsPower; //((32**4)/2)/32 = (32**3)/2
    perGPUKByNumGPUs = PerGPUK_/gpusInK_; //((32**4)/2)/2
    perGPUKByKronRows = PerGPUK_/KronRows_; //
    perGPUNByNumGPUs = PerGPUN_/gpusInK_;
    perGPUNByKronRows = PerGPUN_/KronRows_;
    ColsAByKronRows = ColsA_/KronRows_;
    gcMulUVAColsRatioKronRowsSquare = gc*UVAColsRatioKronRowsSquare;
    ColsCByKronRowsPower = ColsC_/KronRowsPower;
    ColsCByKronCols = ColsC_/KronCols_;
    if (gc == 0) {
      std::cout << "KronCols_ " << KronCols_ << " ColsC_ " << ColsC_ << std::endl
              << "KronRowsPower " << KronRowsPower << std::endl
              << " UVAColsRatioKronRowsSquare " << UVAColsRatioKronRowsSquare << std::endl
              << " perGPUKByNumGPUs " << perGPUKByNumGPUs << std::endl
              << " perGPUKByKronRows " << perGPUKByKronRows << std::endl
              << " perGPUNByNumGPUs " << perGPUNByNumGPUs << std::endl
              << " perGPUNByKronRows " << perGPUNByKronRows << std::endl
              << " ColsAByKronRows " << ColsAByKronRows << std::endl
              << " ColsCByKronRowsPower " << ColsCByKronRowsPower << std::endl
              << " ColsCByKronCols " << ColsCByKronCols << std::endl;
    }
    setGPUResults(0, gpuResults0, gpuResults_);
    setGPUResults(1, gpuResults1, gpuResults_);
    setGPUResults(2, gpuResults2, gpuResults_);
    setGPUResults(3, gpuResults3, gpuResults_);
    setGPUResults(4, gpuResults4, gpuResults_);
    setGPUResults(5, gpuResults5, gpuResults_);
    setGPUResults(6, gpuResults6, gpuResults_);
    setGPUResults(7, gpuResults7, gpuResults_);


  }

  void setGPUResults(int idx, ElemT*& thisResults, ElemT** gpuResults) {
    if (idx < gpusInK)
      thisResults = gpuResults[idx];
    else
      thisResults = nullptr;
  }

  __device__ __forceinline__ ElemT* getLocalGPUResult(uint gc) {
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
  // DistributedParams(const DistributedParams<ElemT, LocalKrons>& x): numGPUs(x.numGPUs),
  //   ColsA(x.ColsA), ColsC(ColsC), storeToDistMems(storeToDistMems) {}

  //   DistributedParams<ElemT, LocalKrons>& operator=(const DistributedParams<ElemT, LocalKrons>& x) {

  //   }
};

#endif