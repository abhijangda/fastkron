#include <cassert>
#include <iostream>
#include <iomanip>

#include "autotuner/autotuner.h"
#include "kmm/kmmalgo.h"
#include "utils/utils.h"

static float minExecTimeOfSeries(KMMProblem problem, uint startKron, bool isDistributed,
                                 TunedKernelsSeries& tunedKernels,
                                 TunedKernelsMap tunedKernelsMap) {
  if (startKron >= problem.n) return 0;
  float minTime = std::numeric_limits<float>::max();
  TunedKernelsSeries minEpilogueKernels;
  TunedKernelFromStart minPrologueKernel;
  auto nextSeries = problem.sub(startKron, problem.n - startKron);

  reverseExecuteGeKMM(nextSeries, nullptr, nullptr, 
               [](const KMMProblem p){return 1;},
    [&](const KMMProblem firstPart, int rstart, void* temps[2], void* r) {
      const int subn = rstart + 1;
      auto tunedProblem = problem.sub(startKron, subn);
      bool isP2P = isDistributed && startKron == 0;
      if (tunedKernelsMap.hasKernel(tunedProblem, isP2P)) {
        TunedKernelsSeries epilogueKernels;
        float kernelTime = tunedKernelsMap.getKernelTime(tunedProblem, isP2P);
        float epilogueTime = minExecTimeOfSeries(problem, startKron + rstart + 1,
                                                 isDistributed, 
                                                 epilogueKernels, tunedKernelsMap);
        if (minTime > kernelTime + epilogueTime) {
          minTime = kernelTime + epilogueTime;
          minEpilogueKernels = epilogueKernels;
          minPrologueKernel = TunedKernelFromStart(tunedKernelsMap.getKernel(tunedProblem, isP2P),
                                                   startKron, startKron + rstart, firstPart.k, kernelTime);
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
                                           cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  //For performance eval we do not need these to contain any value
  
  //Use double buffering for writing result and using output 
  //of previous iteration as input to current
  std::cout << "Fusion enabled?  " << this->fastKron.getUseFusion() << std::endl;
  //A KronMat is a series of SlicedMats
  //We need to get best kernel for all contiguous SlicedMats
  auto err = reverseExecuteGeKMM(problem, nullptr, nullptr, 
               [](const KMMProblem p){return 1;},
  [&](const KMMProblem firstPart, int rstart, void* temps[2], void* r) {
    for (int endP = rstart; endP < problem.n; endP++) {
      auto secondPart = problem.sub(rstart, endP-rstart+1);
      bool distP2PStore = isDistributed && rstart == 0;
      if (tunedKernelsMap.hasKernel(secondPart, distP2PStore)) continue;
      if (!this->fastKron.getUseFusion() and secondPart.n > 1) continue;
      auto bestKernelWithTime = fastKron.kerneldb.tuneKernelForSize(secondPart, distP2PStore, rstart, distParams, stream);
      if (bestKernelWithTime.second < std::numeric_limits<float>::max()) {
        tunedKernelsMap.add(secondPart, distP2PStore,
                            bestKernelWithTime);
      }
    }

    return cudaSuccess;
  });

  return err;
}

cudaError_t Autotuner::tune(KMMProblem problem, cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  float minTime = 0;
  void* temp1_[fastKron.numGPUs_], *temp2_[fastKron.numGPUs_];
  void* Fs[problem.n * fastKron.numGPUs_];
  size_t resultSize = 0, tempSize = 0;
  fastKron.gekmmSizes(problem, &resultSize, &tempSize);
  for (int g = 0; g < fastKron.numGPUs_; g++) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMalloc(&temp1_[g], tempSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&temp2_[g], tempSize * sizeof(float)));

    CUDA_CHECK(cudaMemset(temp1_[g], 1, tempSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(temp2_[g], 1, tempSize * sizeof(float)));

    for (int f = 0; f < problem.n; f++) {
      auto sz = problem.ps[f] * problem.qs[f] * sizeof(float);
      CUDA_CHECK(cudaMalloc(&Fs[g * problem.n + f], sz));
      CUDA_CHECK(cudaMemset(Fs[g * problem.n + f], 1, sz));
    }
  }

  CUDA_CHECK(cudaSetDevice(0));

  if (!fastKron.isDistributed_) {
    //Use temporary as input/output matrix
    auto tmpProblem = KMMProblem(problem, temp1_[0], Fs, temp2_[0]);

    tuneSlicedMulSeries(tmpProblem, false, DistributedParams(), stream);
    std::cout << "Finding min execution time of the series" << std::endl;
    TunedKernelsSeries tunedKernels;
    minTime = minExecTimeOfSeries(problem, 0, false,
                                  tunedKernels, tunedKernelsMap);
    fastKron.tunedKernelSeries = tunedKernels;

  } else {
    if (!checkDistributedKronSizes(problem,
                                   fastKron.perGPUKronBatch_, fastKron.gpusInK_))
      return cudaErrorInvalidValue;

    //In distributed case run every LocalKron series on a single GPU    
    minTime = std::numeric_limits<float>::max();
    uint gpuM, gpuK;
    fastKron.getDistributedSizes(problem.m, problem.k, gpuM, gpuK);
    uint bestMaxLocalKrons = 1;
    TunedKernelsSeries minKernelSeries;
    //For P2P go through all MaxLocalKrons and for NCCL set MaxLocalKrons to maximum value
    int MaxLocalKrons;
    if (fastKron.distComm_ == DistComm::P2P) {
      MaxLocalKrons = 1;
    } else if (fastKron.distComm_ == DistComm::NCCL) {
      if (fastKron.perGPUKronBatch_ > 1)
        MaxLocalKrons = problem.n - 1;
      else
        MaxLocalKrons = 1;
    }

    uint UpperLocalKrons = problem.n;
    if (fastKron.distComm_ == DistComm::NCCL && fastKron.perGPUKronBatch_ == 1)
      UpperLocalKrons = 2;
    
    if (fastKron.gpusInK_ == 1) {
      UpperLocalKrons = problem.n + 1;
      MaxLocalKrons = problem.n;
    }

    //TODO: consider only valid krons 
    for (; MaxLocalKrons < UpperLocalKrons; MaxLocalKrons += 1) {
    float seriesTime = 0;
    TunedKernelsSeries tunedKernelSeries;

    auto tmpProblem = KMMProblem(problem, temp1_[0], Fs, temp2_[0]);
    tmpProblem.m = gpuM;

    for (int i = problem.n - 1; i >= 0; i -= MaxLocalKrons) {
      const uint LocalKrons = std::min(MaxLocalKrons, i + 1);
      //TODO: any way to avoid declaring ps, qs, and fs on stack
      //set max value of N as 64
      auto subproblem = tmpProblem.rsub(i, LocalKrons);
      void** gpuResults = (void**)temp2_;
      DistributedParams distParams(0, 0, fastKron.gpusInK_, subproblem.k, subproblem.l * fastKron.gpusInK_, 
                                   subproblem.k, subproblem.k * fastKron.gpusInK_, subproblem.qs, subproblem.ps, 
                                   subproblem.n);
      distParams.updateGPUResults((void**)gpuResults);
      bool distP2PStore = fastKron.gpusInK_ > 1 && fastKron.isDistributed_ && fastKron.distComm_ == DistComm::P2P;
      tuneSlicedMulSeries(subproblem, distP2PStore, 
                          distParams, stream);
      TunedKernelsSeries tunedKernels;
      seriesTime += minExecTimeOfSeries(subproblem, 0, distP2PStore,
                                        tunedKernels, tunedKernelsMap);
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
    for (int f = 0; f < problem.n; f++) {
      CUDA_CHECK(cudaFree(Fs[g * problem.n + f]));
    }
  }
  
  std::cout <<"Minimum Time " << minTime << " through kernels: " << std::endl;
  for (auto iter = fastKron.tunedKernelSeries.rbegin(); iter != fastKron.tunedKernelSeries.rend(); iter++) {
    std::cout << "  " << (*iter) << std::endl;
    if (fastKron.isDistributed_ and fastKron.gpusInK_ > 1 and 
        ((problem.n - iter->start) % fastKron.perGPUKronBatch_ == 0 or 
        iter->start == 0)) {
      uint gpuM, gpuK;
      fastKron.getDistributedSizes(problem.m, problem.k, gpuM, gpuK);
      std::cout << "  " << "Communicate [" << gpuM << ", " << gpuK << "] among " << 
                   "[GM, " << fastKron.gpusInK_ << "] using " << fastKron.distComm_ << std::endl;
    }
  }
  return cudaSuccess;
}