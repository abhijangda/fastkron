#include <cassert>
#include <iostream>
#include <iomanip>

#include "autotuner.h"
#include "kmmalgo.h"

static float minExecTimeOfSeries(uint M, uint K, const uint NumKronMats, 
                                 uint KronMatCols[], uint KronMatRows[],
                                 uint startKron, bool isDistributed,
                                 TunedKernelsSeries& tunedKernels,
                                 std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>> bestKernels) {
  if (startKron >= NumKronMats) return 0;
  bool distP2PStore = isDistributed;
  float minTime = std::numeric_limits<float>::max();
  TunedKernelsSeries minEpilogueKernels;
  TunedKernelFromStart minPrologueKernel;
  for (uint endKron = startKron; endKron < NumKronMats; endKron++) {
    const uint kronMat = endKron;
    //Include KronMats [startKron, ..., endKron]
    const uint NumFusedKerns = endKron - startKron + 1;
    uint FusedKronMatCols[NumFusedKerns];
    uint FusedKronMatRows[NumFusedKerns];
    for (int k = 0; k < NumFusedKerns; k++) {
      FusedKronMatCols[k] = KronMatCols[kronMat - k];
      FusedKronMatRows[k] = KronMatRows[kronMat - k];
    }

    //TODO: Change tempN to tempK everywhere else
    uint tempK = K;
    for (int reverseKron = NumKronMats - 1; reverseKron > endKron; reverseKron--) {
      tempK = (tempK/KronMatRows[reverseKron])*KronMatCols[reverseKron];
    }

    KronMatmulShape shape = KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                            tempK, M, NumFusedKerns, 
                                            distP2PStore && startKron == 0};
    if (bestKernels.find(shape) == bestKernels.end()) continue;
    auto iter = bestKernels.find(shape);
    TunedKernelsSeries epilogueKernels;
    float kernelTime = iter->second.second;
    float epilogueTime = minExecTimeOfSeries(M, K, NumKronMats, KronMatCols,
                                             KronMatRows, endKron + 1, isDistributed, 
                                             epilogueKernels, bestKernels);
    if (minTime > kernelTime + epilogueTime) {
      minTime = kernelTime + epilogueTime;
      minEpilogueKernels = epilogueKernels;
      minPrologueKernel = TunedKernelFromStart(iter->second.first, 
                                               startKron, endKron, tempK, kernelTime);
    }
  }
  tunedKernels = minEpilogueKernels;
  tunedKernels.push_back(minPrologueKernel);

  assert(minTime < std::numeric_limits<float>::max());
  return minTime;
}

// static float minExecTimeOfSeries(KMMProblem problem, uint startKron, bool isDistributed,
//                                  TunedKernelsSeries& tunedKernels,
//                                  std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>> bestKernels) {
//   if (startKron >= NumKronMats) return 0;
//   bool distP2PStore = isDistributed;
//   float minTime = std::numeric_limits<float>::max();
//   TunedKernelsSeries minEpilogueKernels;
//   TunedKernelFromStart minPrologueKernel;

//   // reverseExecuteGeKMM(problem, nullptr, nullptr, 
//   //              [](const KMMProblem p){return 1;},
//   //   [problem](const KMMProblem firstPart, void* temps[2], void* r) {
//   //     const int subn = p.start - startKron + 1;
//   //     uint qs[problem.shape.n];
//   //     uint ps[problem.shape.n];
//   //     auto secondPart = problem.sub(GeKMMPtrs(), ps, qs, nullptr, p.start, p.n - p.start + 1);
      
//   //     return cudaSuccess;
//   //   });
  
//   for (uint endKron = startKron; endKron < NumKronMats; endKron++) {
//     const uint kronMat = endKron;
//     //Include KronMats [startKron, ..., endKron]
//     const uint NumFusedKerns = endKron - startKron + 1;

//     //TODO: Change tempN to tempK everywhere else
//     uint tempK = K;
//     for (int reverseKron = NumKronMats - 1; reverseKron > endKron; reverseKron--) {
//       tempK = (tempK/KronMatRows[reverseKron])*KronMatCols[reverseKron];
//     }

//     KronMatmulShape shape = KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
//                                             tempK, M, NumFusedKerns, 
//                                             distP2PStore && startKron == 0};
//     if (bestKernels.find(shape) == bestKernels.end()) continue;
//     auto iter = bestKernels.find(shape);
//     TunedKernelsSeries epilogueKernels;
//     float kernelTime = iter->second.second;
//     float epilogueTime = minExecTimeOfSeries(problem, endKron + 1, isDistributed, 
//                                              epilogueKernels, bestKernels);
//     if (minTime > kernelTime + epilogueTime) {
//       minTime = kernelTime + epilogueTime;
//       minEpilogueKernels = epilogueKernels;
//       minPrologueKernel = TunedKernelFromStart(iter->second.first, 
//                                                startKron, endKron, tempK, kernelTime);
//     }
//   }
//   tunedKernels = minEpilogueKernels;
//   tunedKernels.push_back(minPrologueKernel);

//   assert(minTime < std::numeric_limits<float>::max());
//   return minTime;
// }

cudaError_t Autotuner::tuneSlicedMulSeries(const uint NumKronMats, void* x, void* kronMats[],
                              uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                              void* temp1, void* temp2,
                              bool isDistributed, DistributedParams distParams,
                              std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>>& bestKernels,
                              cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  void* kronGemmResults[2] = {(void*)temp1, (void*)temp2};
  //For performance eval we do not need these to contain any value
  void* prevKronResult = kronGemmResults[0];
  void* currKronResult = kronGemmResults[1];
  //TODO: Assumes all factors are of same size and square shape
  // const uint MaxFusedKerns = fastKron.getUseFusion() ? 
  //                            maxFusedKernels(KronMatmulShape{KronMatCols[0], KronMatRows[0], K}) : 1;
  //Use double buffering for writing result and using output 
  //of previous iteration as input to current
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  
  //A KronMat is a series of SlicedMats
  //We need to get best kernel for all contiguous SlicedMats
  for (uint startKron = 0; startKron < NumKronMats; startKron++) {
  for (uint endKron = startKron; endKron < NumKronMats; endKron++) {
    const uint kronMat = endKron;
    //KronMats[startKron, ..., endKron] including endKron
    const uint NumFusedKerns = endKron - startKron + 1;
    void* krons[NumFusedKerns];
    uint FusedKronMatCols[NumFusedKerns];
    uint FusedKronMatRows[NumFusedKerns];
    for (int k = 0; k < NumFusedKerns; k++) {
      krons[k] = kronMats[kronMat - k];
      FusedKronMatCols[k] = KronMatCols[kronMat - k];
      FusedKronMatRows[k] = KronMatRows[kronMat - k];
    }
    uint tempN = K;
    for (int reverseKron = NumKronMats - 1; reverseKron > endKron; reverseKron--) {
      tempN = (tempN/KronMatRows[reverseKron])*KronMatCols[reverseKron];
    }
    uint outTempN = (tempN/KronMatRows[endKron])*KronMatCols[endKron];
    // std::cout << "endKron " << endKron << " startKron " << startKron << " tempN " << tempN << std::endl;
    bool distP2PStore = isDistributed && startKron == 0;
    cudaError_t status;
    KronMatmulShape shape = KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                            tempN, M, NumFusedKerns, distP2PStore};
    if (bestKernels.find(shape) != bestKernels.end()) {
      continue;
    }
    if (!fastKron.getUseFusion() and NumFusedKerns > 1) continue;
    KernelInfo bestKernel;
    float minTime = std::numeric_limits<float>::max();
    const uint runs = 5;
    const uint warmups = 2;
    std::cout << "Tuning for shape "  << shape << std::endl;
    for (auto shapeAndKernels : fastKron.compiledKernels) {
      if (!shapeAndKernels.first.sameKronSize(shape)) continue;
      for (auto kernel : shapeAndKernels.second) {
        if (!kernel.canCompute(shape)) continue;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int r = 0; r < warmups + runs; r++) {
          if (r == warmups) CUDA_CHECK(cudaEventRecord(start, stream));
          if (distP2PStore) {
            status = fastKron.kernelInvoker.fusedDistributedSlicedMatmul(NumFusedKerns, kernel, endKron, (void*)prevKronResult, 
                                                  (void**)krons, (void*)currKronResult, M, outTempN, tempN, 
                                                  FusedKronMatCols, FusedKronMatRows, 
                                                  distParams, EpilogueParams::create<float>(), stream);
          } else {
            status = fastKron.kernelInvoker.fusedSlicedMatmul(NumFusedKerns, kernel, endKron, (void*)prevKronResult,
                                       (void**)krons, (void*)currKronResult, M, outTempN, tempN, 
                                       FusedKronMatCols, FusedKronMatRows,
                                       EpilogueParams::create<float>(), stream);
          }
          // if (status != cudaSuccess) break;
        }
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        
        if (status != cudaSuccess)
          std::cout << "Error: " << cudaGetErrorString(status) << " for " << kernel << " tempN " << tempN << std::endl;
        float kernelTime;
        CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, end));
        std::cout << std::fixed << std::setprecision(2) << 
                     kernel << " runs in " << (kernelTime/runs) << " ms " << std::endl;
        if (kernelTime < minTime) {
          bestKernel = kernel;
          minTime = kernelTime;
        }
        if (status != cudaSuccess) return status;
      }
    }

    if (minTime < std::numeric_limits<float>::max()) {
      std::cout << std::fixed << std::setprecision(2) <<
                   "Best kernel for " << shape << ": " << bestKernel << " runs in " << (minTime/runs) << " ms" << std::endl;
      bestKernels.emplace(std::make_pair(shape, std::make_pair(bestKernel, minTime/runs)));
    }
  }}

  return cudaSuccess;
}

cudaError_t Autotuner::tune(const uint NumKronMats, void* x, void** kronMats, 
                     uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                     cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return cudaErrorInvalidValue;

  std::cout << "N " << N << " K " << K << " KronMatCols[0] " << KronMatCols[0] << " KronMatRows[0] " << KronMatRows[0] << std::endl;
  float minTime = 0;
  void* temp1_[fastKron.numGPUs_], *temp2_[fastKron.numGPUs_];
  size_t resultSize = 0, tempSize = 0;
  gekmmSizes(&fastKron, NumKronMats, M, N, K, KronMatCols, KronMatRows, 
             &resultSize, &tempSize);
  std::cout << "172: " << tempSize << std::endl;
  for (int g = 0; g < fastKron.numGPUs_; g++) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMalloc(&temp1_[g], tempSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&temp2_[g], tempSize * sizeof(float)));
  }

  if (!fastKron.isDistributed_) {
    CUDA_CHECK(cudaSetDevice(0));
    std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>> bestKernels;
    tuneSlicedMulSeries(NumKronMats, x, kronMats, M, N, K, KronMatCols, KronMatRows, 
                      (void*)temp1_[0], (void*)temp2_[0], false, DistributedParams(), 
                      bestKernels, stream);
    std::cout << "Finding min execution time of the series" << std::endl;
    TunedKernelsSeries tunedKernels;
    minTime = minExecTimeOfSeries(M, K, NumKronMats,
                                  KronMatCols, KronMatRows, 0, false,
                                  tunedKernels, bestKernels);
    fastKron.tunedKernelSeries = tunedKernels;

  } else {
    if (!checkDistributedKronSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows, 
                                   fastKron.perGPUKronBatch_, fastKron.gpusInK_))
      return cudaErrorInvalidValue;

    //In distributed case run every LocalKron series on a single GPU    
    CUDA_CHECK(cudaSetDevice(0));
    minTime = std::numeric_limits<float>::max();
    uint gpuM, gpuK;
    fastKron.getDistributedSizes(M, K, gpuM, gpuK);
    uint prevTempN = gpuK;
    //TODO: This loop is really common and should be a macro?
    std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>> bestKernels;

    uint bestMaxLocalKrons = 1;
    TunedKernelsSeries minKernelSeries;
    //For P2P go through all MaxLocalKrons and for NCCL set MaxLocalKrons to maximum value
    uint MaxLocalKrons;
    if (fastKron.distComm_ == DistComm::P2P) {
      MaxLocalKrons = 1;
    } else if (fastKron.distComm_ == DistComm::NCCL) {
      if (fastKron.perGPUKronBatch_ > 1)
        MaxLocalKrons = NumKronMats - 1;
      else
        MaxLocalKrons = 1;
    }
    uint UpperLocalKrons = NumKronMats;
    if (fastKron.distComm_ == DistComm::NCCL && fastKron.perGPUKronBatch_ == 1)
      UpperLocalKrons = 2;
    
    if (fastKron.gpusInK_ == 1)
      UpperLocalKrons = 2;

    //TODO: consider only valid krons 
    for (; MaxLocalKrons < UpperLocalKrons; MaxLocalKrons += 1) {
    uint seriesTime = 0;
    TunedKernelsSeries tunedKernelSeries;
    
    for (uint i = 0; i < NumKronMats; i += MaxLocalKrons) {
      const uint kronMat = NumKronMats - i - 1;
      const uint LocalKrons = std::min(MaxLocalKrons, NumKronMats - i);
      uint currTempN = prevTempN;
      uint LocalKronMatCols[LocalKrons];
      uint LocalKronMatRows[LocalKrons];
      for (int k = 0; k < LocalKrons; k++) {
        LocalKronMatCols[k] = KronMatCols[kronMat - k];
        LocalKronMatRows[k] = KronMatRows[kronMat - k];
        currTempN = (currTempN/LocalKronMatRows[k])*LocalKronMatCols[k];
      }

      void** gpuResults = (void**)temp2_;
      int prevFullK = prevTempN * fastKron.gpusInK_;
      int currFullN = currTempN * fastKron.gpusInK_;
      DistributedParams distParams(0, 0, fastKron.gpusInK_, prevFullK, currFullN, 
                                      prevFullK, currFullN, LocalKronMatCols, LocalKronMatRows, LocalKrons);
      distParams.updateGPUResults((void**)gpuResults);
      tuneSlicedMulSeries(LocalKrons, x, kronMats, gpuM, currTempN, prevTempN, 
                        LocalKronMatCols, LocalKronMatRows, temp1_[0], temp2_[0],
                        fastKron.gpusInK_ > 1 && fastKron.isDistributed_ && fastKron.distComm_ == DistComm::P2P, 
                        distParams, bestKernels, stream);
      TunedKernelsSeries tunedKernels;
      seriesTime += minExecTimeOfSeries(gpuM, prevTempN, LocalKrons,
                                     LocalKronMatCols, LocalKronMatRows, 0,
                                     fastKron.gpusInK_ > 1 &&fastKron.isDistributed_ && fastKron.distComm_ == DistComm::P2P,
                                     tunedKernels, bestKernels);

      for (auto tunedKernel : tunedKernels) {
        tunedKernel.start += kronMat + 1 - LocalKrons;
        tunedKernel.end   += kronMat + 1 - LocalKrons;
        tunedKernelSeries.insert(tunedKernelSeries.begin(), tunedKernel);
      }
    }
    
    if (seriesTime < minTime) {
      minTime = seriesTime;
      fastKron.tunedKernelSeries = tunedKernelSeries;
      fastKron.perGPUKronBatch_ = MaxLocalKrons;
    }
    }
  }

  for (int g = 0; g < fastKron.numGPUs_; g++) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaFree(temp1_[g]));
    CUDA_CHECK(cudaFree(temp2_[g]));
  }

  std::cout <<"Minimum Time " << minTime << " through kernels: " << std::endl;
  for (auto iter = fastKron.tunedKernelSeries.rbegin(); iter != fastKron.tunedKernelSeries.rend(); iter++) {
    std::cout << "  " << (*iter) << std::endl;
  }
  return cudaSuccess;
}