#include <cassert>
#include <iostream>
#include <iomanip>

#include "autotuner/autotuner.h"
#include "kmm/kmmalgo.h"
#include "utils/utils.h"

static float minExecTimeOfSeries(KMMProblem problem, uint startKron, bool isDistributed,
                                 TunedKernelsSeries& tunedKernels,
                                 TunedKernelsMap tunedKernelsMap) {
  if (startKron >= problem.n()) return 0;
  float minTime = std::numeric_limits<float>::max();
  TunedKernelsSeries minEpilogueKernels;
  TunedKernelFromStart minPrologueKernel;
  auto nextSeries = problem.sub(startKron, problem.n() - startKron);

  reverseExecuteGeKMM(nextSeries, nullptr, Matrix(), 
               [](const KMMProblem p){return 1;},
  [&](const KMMProblem firstPart, int rstart, void* temps[2], Matrix result) {
    const int subn = rstart + 1;
    auto tunedProblem = problem.sub(startKron, subn);
    if (rstart != problem.n() - 1) tunedProblem.setOpX(fastKronOp_N);
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
                                                 startKron, startKron + rstart, firstPart.k(), kernelTime);
      }
    }

    return cudaSuccess;
  });

  tunedKernels = minEpilogueKernels;
  tunedKernels.push_back(minPrologueKernel);
  assert(minTime < std::numeric_limits<float>::max());
  return minTime;
}

cudaError_t Autotuner::tune(KMMProblem problem,
                            bool isDistributed, DistributedParams distParams,
                            cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  //For performance eval we do not need these to contain any value
  
  //Use double buffering for writing result and using output 
  //of previous iteration as input to current
  std::cout << "Fusion enabled?  " << this->fastKron.getUseFusion() << std::endl;
  //A KronMat is a series of SlicedMats
  //We need to get best kernel for all contiguous SlicedMats
  auto err = reverseExecuteGeKMM(problem, nullptr, Matrix(), 
               [](const KMMProblem p){return 1;},
  [&](const KMMProblem firstPart, int rstart, void* temps[2], Matrix r) {
    for (int endP = rstart; endP < problem.n(); endP++) {
      auto secondPart = problem.sub(rstart, endP-rstart+1);
      if (endP != 0) secondPart.setOpX(fastKronOp_N);
      bool distP2PStore = isDistributed && rstart == 0;
      if (tunedKernelsMap.hasKernel(secondPart, distP2PStore)) continue;
      if (!this->fastKron.getUseFusion() and secondPart.n() > 1) continue;
      auto bestKernelWithTime = fastKron.kerneldb.tuneKernelForProblem(secondPart, distP2PStore, rstart, distParams, stream);
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
  Matrix result, temp;
  fastKron.gekmmResultTemp(problem, result, temp);
  
  Matrix temp1[fastKron.numGPUs_];
  Matrix temp2[fastKron.numGPUs_];
//TODO: Make this FactorArray
  Factor Fs[fastKron.numGPUs_][problem.n()];

  for (uint32_t p = 0; p < fastKron.numGPUs_; p++) {
    //TODO: Init temp to 1
    fastKron.gekmmResultTemp(problem, result, temp1[p]);
    fastKron.gekmmResultTemp(problem, result, temp2[p]);
    fastKron.kerneldb.procMalloc(p, temp1[p]);
    fastKron.kerneldb.procMalloc(p, temp2[p]);
    fastKron.kerneldb.procMemset(p, temp1[p], 1.0f);
    fastKron.kerneldb.procMemset(p, temp2[p], 1.0f);

    for (int f = 0; f < problem.n(); f++) {
      Fs[p][f] = problem.f(f);
      fastKron.kerneldb.procMalloc(p, Fs[p][f]);
      fastKron.kerneldb.procMemset(p, Fs[p][f], 1.0f);
    }
  }

  CUDA_CHECK(cudaSetDevice(0));

  // if (true) {
  //   float* tt = new float[8 * 16384];
  //   CUDA_CHECK(cudaMemcpy(tt, temp1[0].data(), 8*16384*sizeof(float), cudaMemcpyDeviceToHost));
  //   printf("162: %p\n", temp1[0].data());
  //   for (int i = 0; i < 8; i++) {
  //     for (int j = 0; j < 16384; j++) {
  //       if (i == 0) //if (tt[i * 16384 + j] != 0.0f) printf("tt[%d * 16384 + %d] %f\n", i, j, tt[i * 16384 + j]);
  //       printf("%f\n", tt[i * 16384 + j]);
  //     }
  //   }
  // }

  if (!fastKron.isDistributed_) {
    //Use temporary as input/output matrix
    //TODO: fix this
    auto tmpProblem = KMMProblem(Matrix(problem.x().m(), problem.x().n(), temp1[0].data()), 
                                 problem.opX(), problem.n(), &Fs[0][0], problem.opFs(),
                                 Matrix(problem.y().m(), problem.y().n(), temp2[0].data()));
    tune(tmpProblem, false, DistributedParams(), stream);
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
    fastKron.getDistributedSizes(problem.m(), problem.k(), gpuM, gpuK);
    uint bestMaxLocalKrons = 1;
    TunedKernelsSeries minKernelSeries;
    //For P2P go through all MaxLocalKrons and for NCCL set MaxLocalKrons to maximum value
    int MaxLocalKrons;
    if (fastKron.distComm_ == DistComm::P2P) {
      MaxLocalKrons = 1;
    } else if (fastKron.distComm_ == DistComm::NCCL) {
      if (fastKron.perGPUKronBatch_ > 1)
        MaxLocalKrons = problem.n() - 1;
      else
        MaxLocalKrons = 1;
    }

    uint UpperLocalKrons = problem.n();
    if (fastKron.distComm_ == DistComm::NCCL && fastKron.perGPUKronBatch_ == 1)
      UpperLocalKrons = 2;
    
    if (fastKron.gpusInK_ == 1) {
      UpperLocalKrons = problem.n() + 1;
      MaxLocalKrons = problem.n();
    }

    //TODO: consider only valid krons 
    for (; MaxLocalKrons < UpperLocalKrons; MaxLocalKrons += 1) {
    float seriesTime = 0;
    TunedKernelsSeries tunedKernelSeries;

    auto tmpProblem = KMMProblem(Matrix(gpuM, gpuK, temp1[0].data()), 
                                 problem.opX(), problem.n(), &Fs[0][0], problem.opFs(),
                                 Matrix(gpuM, problem.y().n()/fastKron.gpusInK_, temp2[0].data()));

    for (int i = problem.n() - 1; i >= 0; i -= MaxLocalKrons) {
      const uint LocalKrons = std::min(MaxLocalKrons, i + 1);
      //TODO: any way to avoid declaring ps, qs, and fs on stack
      //set max value of N as 64
      auto subproblem = tmpProblem.rsub(i, LocalKrons);
      void* gpuResults[fastKron.numGPUs_] = {nullptr};
      std::transform(temp2, temp2 + fastKron.numGPUs_, &gpuResults[0], [](Matrix m) {return m.data();});
      DistributedParams distParams(0, 0, fastKron.gpusInK_, subproblem.k() * fastKron.gpusInK_, subproblem.l() * fastKron.gpusInK_, 
                                   subproblem.k(), subproblem.l(), subproblem.fs(), 
                                   subproblem.n());
      distParams.updateGPUResults((void**)gpuResults);
      bool distP2PStore = fastKron.gpusInK_ > 1 && fastKron.isDistributed_ && fastKron.distComm_ == DistComm::P2P;
      tune(subproblem, distP2PStore, distParams, stream);
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

  for (int p = 0; p < fastKron.numGPUs_; p++) {
    fastKron.kerneldb.procFree(p, temp1[p]);
    fastKron.kerneldb.procFree(p, temp2[p]);
    for (int f = 0; f < problem.n(); f++) {
      //TODO: // CUDA_CHECK(cudaFree(Fs[g * problem.n() + f]));
    }
  }
  
  std::cout <<"Minimum Time " << minTime << " through kernels: " << std::endl;
  for (auto iter = fastKron.tunedKernelSeries.rbegin(); iter != fastKron.tunedKernelSeries.rend(); iter++) {
    std::cout << "  " << (*iter) << std::endl;
    if (fastKron.isDistributed_ and fastKron.gpusInK_ > 1 and 
        ((problem.n() - iter->start) % fastKron.perGPUKronBatch_ == 0 or 
        iter->start == 0)) {
      uint gpuM, gpuK;
      fastKron.getDistributedSizes(problem.m(), problem.k(), gpuM, gpuK);
      std::cout << "  " << "Communicate [" << gpuM << ", " << gpuK << "] among " << 
                   "[GM, " << fastKron.gpusInK_ << "] using " << fastKron.distComm_ << std::endl;
    }
  }
  return cudaSuccess;
}