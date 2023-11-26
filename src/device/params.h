#include <iostream>
#include <numeric>

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

template<typename ElemT>
struct EpilogueParams {
  const ElemT * __restrict__ glD;
  const ElemT alpha;
  const ElemT beta;
  EpilogueParams() : alpha((ElemT)1.0f), beta(0), glD(nullptr) {}
  EpilogueParams(const ElemT alpha, const ElemT beta, const ElemT* glD) :
    glD(glD), alpha(alpha), beta(beta) {}
};

template<uint NumFusedKerns>
struct KernelParams {
  const uint RowsC;
  const uint ColsC; //TODO: Change to LocalColsC
  const uint ColsA;
  uint KronRows[NumFusedKerns];
  uint KronCols[NumFusedKerns];
  const void * __restrict__ glA;
  const void * __restrict__ glKronMats[NumFusedKerns];
  void       * __restrict__ glC;
  const uint kp_idx;

  KernelParams(const uint RowsC, const uint ColsC, const uint ColsA,
               const uint KronRows[NumFusedKerns], const uint KronCols[NumFusedKerns], const void* glA,
               void* glKronMats[NumFusedKerns], void* glC, uint kp_idx) :
               RowsC(RowsC), ColsC(ColsC), ColsA(ColsA), glA(glA), glC(glC), kp_idx(kp_idx) {
    for (int i = 0; i < NumFusedKerns; i++) {
      this->KronRows[i] = KronRows[i];
      this->KronCols[i] = KronCols[i];
      this->glKronMats[i] = glKronMats[i];
    }
  }
};


template<uint NumFusedKerns>
struct FusedParams {
  uint KronColsPower;
  uint UVAColsRatioKronColsSquare;
  uint ColsCByKronColsPower;
  
  FusedParams(const uint RowsC, const uint ColsC, const uint ColsA, const uint TileSizeColsA,
              const uint KronRows[NumFusedKerns], const uint KronCols[NumFusedKerns]) {
    KronColsPower = power(KronCols[0], NumFusedKerns);
    UVAColsRatioKronColsSquare = TileSizeColsA/KronColsPower;
    ColsCByKronColsPower = ColsC/KronColsPower;
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
                    const uint KronCols_[], const uint KronRows_[], const uint LocalKrons_) :
    gr(gr_), gc(gc_), gpusInK(gpusInK_), ColsA(ColsA_), ColsC(ColsC_),
    LocalKrons(LocalKrons_) {
    
    const uint KronRowsPower = std::reduce(KronRows_, KronRows_ + LocalKrons_, 1, std::multiplies<uint>());
    const uint KronColsPower = std::reduce(KronCols_, KronCols_ + LocalKrons_, 1, std::multiplies<uint>());
    UVAColsRatioKronRowsSquare = PerGPUN_/KronColsPower; //
    perGPUKByNumGPUs = PerGPUK_/gpusInK_; //
    perGPUNByNumGPUs = PerGPUN_/gpusInK_;
    perGPUNByKronCols = PerGPUN_/KronCols_[LocalKrons_-1];
    gcMulUVAColsRatioKronRowsSquare = gc*UVAColsRatioKronRowsSquare;
    ColsCByKronCols = ColsC_/KronCols_[LocalKrons_-1];
    ColsCByKronColsPower = ColsC_/KronColsPower;
    perGPUNByKronRows = PerGPUN_/KronRows_[LocalKrons_-1];
    perGPUKByKronRows = PerGPUK_/KronRows_[LocalKrons_-1];
    ColsCByKronRowsPower = ColsC_/KronRowsPower;
    ColsAByKronRows = ColsA_/KronRows_[LocalKrons_-1];
    // if (gc == 0) {
    //   std::cout << "KronCols_ " << KronCols_ << " ColsC_ " << ColsC_ << std::endl
    //           << "KronRowsPower " << KronRowsPower << std::endl
    //           << " UVAColsRatioKronRowsSquare " << UVAColsRatioKronRowsSquare << std::endl
    //           << " perGPUKByNumGPUs " << perGPUKByNumGPUs << std::endl
    //           << " perGPUKByKronRows " << perGPUKByKronRows << std::endl
    //           << " perGPUNByNumGPUs " << perGPUNByNumGPUs << std::endl
    //           << " perGPUNByKronRows " << perGPUNByKronRows << std::endl
    //           << " ColsAByKronRows " << ColsAByKronRows << std::endl
    //           << " ColsCByKronRowsPower " << ColsCByKronRowsPower << std::endl
    //           << " ColsCByKronCols " << ColsCByKronCols << std::endl;
    // }
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

  void setGPUResults(int idx, void*& thisResults, void** gpuResults) {
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

#endif