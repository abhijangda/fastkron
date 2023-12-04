#include <cassert>
#include <iostream>
#include <iomanip>

#include "autotuner.h"
#include "kmmalgo.h"

static float minExecTimeOfSeries(KMMProblem problem, uint startKron, bool isDistributed,
                                 TunedKernelsSeries& tunedKernels,
                                 std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>> bestKernels) {
  if (startKron >= problem.n) return 0;
  bool distP2PStore = isDistributed;
  float minTime = std::numeric_limits<float>::max();
  TunedKernelsSeries minEpilogueKernels;
  TunedKernelFromStart minPrologueKernel;
  uint qs[problem.n];
  uint ps[problem.n];
  auto nextSeries = problem.sub(ps, qs, nullptr, 
                                startKron, problem.n - startKron);

  reverseExecuteGeKMM(nextSeries, nullptr, nullptr, 
               [](const KMMProblem p){return 1;},
    [&](const KMMProblem firstPart, void* temps[2], void* r) {
      const int subn = firstPart.rstart + 1;
      uint qs[problem.n];
      uint ps[problem.n];
      
      KronMatmulShape shape = KronMatmulShape{firstPart.qs[0], firstPart.ps[0], 
                                              firstPart.k, problem.m, subn, 
                                              distP2PStore && startKron == 0};
      if (bestKernels.find(shape) != bestKernels.end()) {
        auto iter = bestKernels.find(shape);
        TunedKernelsSeries epilogueKernels;
        float kernelTime = iter->second.second;
        float epilogueTime = minExecTimeOfSeries(problem, startKron + firstPart.rstart + 1,
                                                isDistributed, 
                                                epilogueKernels, bestKernels);
        if (minTime > kernelTime + epilogueTime) {
          minTime = kernelTime + epilogueTime;
          minEpilogueKernels = epilogueKernels;
          minPrologueKernel = TunedKernelFromStart(iter->second.first, 
                                                  startKron, startKron + firstPart.rstart, firstPart.k, kernelTime);
        }
      }

      return cudaSuccess;
    });

  tunedKernels = minEpilogueKernels;
  tunedKernels.push_back(minPrologueKernel);
  assert(minTime < std::numeric_limits<float>::max());
  return minTime;
}

cudaError_t Autotuner::tuneSlicedMulSeries(KMMProblem problem,
                                           bool isDistributed, DistributedParams distParams,
                                           std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>>& bestKernels,
                                           cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  //For performance eval we do not need these to contain any value
  
  //Use double buffering for writing result and using output 
  //of previous iteration as input to current
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  std::cout << "Fusion enabled?  " << this->fastKron.getUseFusion() << std::endl;
  //A KronMat is a series of SlicedMats
  //We need to get best kernel for all contiguous SlicedMats
  auto err = reverseExecuteGeKMM(problem, nullptr, nullptr, 
               [](const KMMProblem p){return 1;},
  [&](const KMMProblem firstPart, void* temps[2], void* r) {
    for (int endP = firstPart.rstart; endP < problem.n; endP++) {
      uint qs[problem.n];
      uint ps[problem.n];
      void* fs[problem.n];
      
      auto secondPart = problem.sub(ps, qs, fs, firstPart.rstart, endP-firstPart.rstart+1);
      bool distP2PStore = isDistributed && firstPart.rstart == 0;
      KronMatmulShape shape = KronMatmulShape{secondPart.qs[0], secondPart.ps[0], 
                                              secondPart.k, secondPart.m, secondPart.n, distP2PStore};
      if (bestKernels.find(shape) != bestKernels.end()) continue;
      if (!this->fastKron.getUseFusion() and secondPart.n > 1) continue;
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
          cudaError_t status;
          for (int r = 0; r < warmups + runs; r++) {
            if (r == warmups) CUDA_CHECK(cudaEventRecord(start, stream));
            if (distP2PStore) {
              status = fastKron.kernelInvoker.fusedDistributedSlicedMatmul(secondPart.n, kernel, firstPart.rstart + secondPart.rstart,
                                                    secondPart.x,
                                                    secondPart.fs, secondPart.y, secondPart.m, 1, secondPart.k, 
                                                    secondPart.qs, secondPart.ps,
                                                    distParams, EpilogueParams::create<float>(), stream);
            } else {
              status = fastKron.kernelInvoker.fusedSlicedMatmul(secondPart.n, kernel, firstPart.rstart + secondPart.rstart,
                                                                secondPart.x, 
                                                                secondPart.fs, secondPart.y, secondPart.m, secondPart.l, secondPart.k, 
                                                                secondPart.qs, secondPart.ps,
                                                                EpilogueParams::create<float>(), stream);
            }
            // if (status != cudaSuccess) break;
          }
          CUDA_CHECK(cudaEventRecord(end, stream));
          CUDA_CHECK(cudaEventSynchronize(end));
          
          if (status != cudaSuccess)
            std::cout << "Error: " << cudaGetErrorString(status) << " for " << kernel << " K " << secondPart.k << std::endl;
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
    }
    
    return cudaSuccess;
  });

  return err;
}

cudaError_t Autotuner::tune(const uint NumKronMats, void* x, void** kronMats, 
                     uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                     cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return cudaErrorInvalidValue;

  float minTime = 0;
  void* temp1_[fastKron.numGPUs_], *temp2_[fastKron.numGPUs_];
  size_t resultSize = 0, tempSize = 0;
  gekmmSizes(&fastKron, M, NumKronMats, KronMatRows, KronMatCols, 
             &resultSize, &tempSize);
  for (int g = 0; g < fastKron.numGPUs_; g++) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMalloc(&temp1_[g], tempSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&temp2_[g], tempSize * sizeof(float)));
  }

  std::unordered_map<KronMatmulShape, std::pair<KernelInfo, float>> bestKernels;

  if (!fastKron.isDistributed_) {
    CUDA_CHECK(cudaSetDevice(0));
    auto problem = KMMProblem(M, NumKronMats, KronMatRows, KronMatCols, 
                              temp1_[0], kronMats, temp2_[0]);
    tuneSlicedMulSeries(problem, false, DistributedParams(), 
                      bestKernels, stream);
    std::cout << "Finding min execution time of the series" << std::endl;
    TunedKernelsSeries tunedKernels;
    minTime = minExecTimeOfSeries(problem, 0, false,
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
    uint bestMaxLocalKrons = 1;
    TunedKernelsSeries minKernelSeries;
    //For P2P go through all MaxLocalKrons and for NCCL set MaxLocalKrons to maximum value
    int MaxLocalKrons;
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
    
    if (fastKron.gpusInK_ == 1) {
      UpperLocalKrons = NumKronMats + 1;
      MaxLocalKrons = NumKronMats;
    }

    //TODO: consider only valid krons 
    for (; MaxLocalKrons < UpperLocalKrons; MaxLocalKrons += 1) {
    float seriesTime = 0;
    TunedKernelsSeries tunedKernelSeries;

    KMMProblem problem(gpuM, NumKronMats, KronMatRows, KronMatCols,
                       temp1_[0], kronMats, temp2_[0]);
    for (int i = problem.n - 1; i >= 0; i -= MaxLocalKrons) {
      const uint LocalKrons = std::min(MaxLocalKrons, i + 1);
      //TODO: any way to avoid declaring ps, qs, and fs on stack
      uint ps[LocalKrons];
      uint qs[LocalKrons];
      void* fs[LocalKrons];
      auto subproblem = problem.rsub(ps, qs, fs, i, LocalKrons);
      void** gpuResults = (void**)temp2_;
      DistributedParams distParams(0, 0, fastKron.gpusInK_, subproblem.k, subproblem.l * fastKron.gpusInK_, 
                                   subproblem.k, subproblem.k * fastKron.gpusInK_, subproblem.qs, subproblem.ps, 
                                   subproblem.n);
      distParams.updateGPUResults((void**)gpuResults);
      bool distP2PStore = fastKron.gpusInK_ > 1 && fastKron.isDistributed_ && fastKron.distComm_ == DistComm::P2P;
      tuneSlicedMulSeries(subproblem, distP2PStore, 
                          distParams, bestKernels, stream);
      TunedKernelsSeries tunedKernels;
      seriesTime += minExecTimeOfSeries(subproblem, 0, distP2PStore,
                                        tunedKernels, bestKernels);
      for (auto tunedKernel : tunedKernels) {
        tunedKernel.start += i + 1 - LocalKrons;
        tunedKernel.end   += i + 1 - LocalKrons;
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
    if (fastKron.isDistributed_ and fastKron.gpusInK_ > 1 and 
        ((NumKronMats - iter->start) % fastKron.perGPUKronBatch_ == 0 or 
        iter->start == 0)) {
      uint gpuM, gpuK;
      fastKron.getDistributedSizes(M, K, gpuM, gpuK);
      std::cout << "  " << "Communicate [" << gpuM << ", " << gpuK << "] among " << 
                   "[GM, " << fastKron.gpusInK_ << "] using " << fastKron.distComm_ << std::endl;
    }
  }
  return cudaSuccess;
}