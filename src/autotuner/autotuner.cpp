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
    if (problem.opX() == fastKronOp_T && startKron + subn == problem.n()) {
      //If opX is T and the tunedProblem has reached end of the problem
      //then consider only TT or TN kernels
      tunedProblem.setOpX(fastKronOp_T);
    } else {
      tunedProblem.setOpX(fastKronOp_N);
    }
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

    return fastKronSuccess;
  });

  tunedKernels = minEpilogueKernels;
  tunedKernels.push_back(minPrologueKernel);
  assert(minTime < std::numeric_limits<float>::max());
  return minTime;
}

fastKronError Autotuner::tune(KMMProblem problem,
                            bool isDistributed, DistributedParams distParams) {
  //Only row major layout of all matrics is supported.
  //For performance eval we do not need these to contain any value
  
  //Use double buffering for writing result and using output 
  //of previous iteration as input to current
  //A KronMat is a series of SlicedMats
  //We need to get best kernel for all contiguous SlicedMats

  auto kernelDb = fastKron.getBackendKernelDb();

  auto err = reverseExecuteGeKMM(problem, nullptr, Matrix(), 
               [](const KMMProblem p){return 1;},
  [&](const KMMProblem firstPart, int rstart, void* temps[2], Matrix r) {
    for (int endP = rstart; endP < problem.n(); endP++) {
      auto secondPart = problem.sub(rstart, endP-rstart+1);
      if (rstart + secondPart.n() < problem.n()) secondPart.setOpX(fastKronOp_N);
      bool distP2PStore = isDistributed && rstart == 0;
      if (tunedKernelsMap.hasKernel(secondPart, distP2PStore)) continue;
      if (!this->fastKron.getUseFusion() and secondPart.n() > 1) continue;
      auto bestKernelWithTime = kernelDb->tuneKernelForProblem(secondPart, distP2PStore, rstart, distParams);
      if (bestKernelWithTime.second < std::numeric_limits<float>::max()) {
        tunedKernelsMap.add(secondPart, distP2PStore,
                            bestKernelWithTime);
      }
    }

    return fastKronSuccess;
  });

  return err;
}

fastKronError Autotuner::tune(KMMProblem problem) {
  //Only row major layout of all matrics is supported.
  if (!env::getTune()) return fastKronSuccess;

  float minTime = 0;
  Matrix result, temp;
  fastKron.gekmmResultTemp(problem, result, temp);
  uint devicesPerProc = 1;

  switch (fastKron.backend) {
    case fastKronBackend_CUDA:
#ifdef ENABLE_CUDA
    devicesPerProc = fastKron.cudaKernels.numGPUs_;
#endif
    break;
    case fastKronBackend_X86:
      devicesPerProc = 1;
      break;
  }

  auto kernelDb = fastKron.getBackendKernelDb();

  Matrix temp1[devicesPerProc];
  Matrix temp2[devicesPerProc];
  //TODO: Make this FactorArray
  Factor Fs[devicesPerProc][problem.n()];
  for (uint32_t p = 0; p < devicesPerProc; p++) {
    //TODO: Init temp to 1
    fastKron.gekmmResultTemp(problem, result, temp1[p]);
    fastKron.gekmmResultTemp(problem, result, temp2[p]);
    kernelDb->procMalloc(p, temp1[p]);
    kernelDb->procMalloc(p, temp2[p]);
    kernelDb->procMemset(p, temp1[p], 1.0f);
    kernelDb->procMemset(p, temp2[p], 1.0f);

    for (int f = 0; f < problem.n(); f++) {
      Fs[p][f] = problem.f(f);
      kernelDb->procMalloc(p, Fs[p][f]);
      kernelDb->procMemset(p, Fs[p][f], 1.0f);
    }
  }

  kernelDb->initTune();

  if (devicesPerProc <= 1) {
    //Use temporary as input/output matrix
    //TODO: fix this
    auto tmpProblem = KMMProblem(problem.type(), Matrix(problem.x().m(), problem.x().n(), temp1[0].data()), 
                                 problem.opX(), problem.n(), &Fs[0][0], problem.opFs(),
                                 Matrix(problem.y().m(), problem.y().n(), temp2[0].data()));
    tune(tmpProblem, false, DistributedParams());
    std::cout << "Finding min execution time of the series" << std::endl;
    TunedKernelsSeries tunedKernels;
    minTime = minExecTimeOfSeries(problem, 0, false,
                                  tunedKernels, tunedKernelsMap);
    fastKron.tunedKernelSeries = tunedKernels;
  } else {
#ifdef ENABLE_CUDA
    assert(fastKron.backend == fastKronBackend_CUDA);
    assert(fastKron.cudaKernels.isDistributed_ == true);
    if (!checkDistributedKronSizes(problem,
                                   fastKron.cudaKernels.perGPUKronBatch_, fastKron.cudaKernels.gpusInK_))
      return fastKronInvalidKMMProblem;

    //In distributed case run every LocalKron series on a single GPU    
    minTime = std::numeric_limits<float>::max();
    uint gpuM, gpuK;
    fastKron.getDistributedSizes(problem.m(), problem.k(), gpuM, gpuK);
    uint bestMaxLocalKrons = 1;
    TunedKernelsSeries minKernelSeries;
    //For P2P go through all MaxLocalKrons and for NCCL set MaxLocalKrons to maximum value
    int MaxLocalKrons;
    if (fastKron.cudaKernels.distComm_ == DistComm::P2P) {
      MaxLocalKrons = 1;
    } else if (fastKron.cudaKernels.distComm_ == DistComm::NCCL) {
      if (fastKron.cudaKernels.perGPUKronBatch_ > 1)
        MaxLocalKrons = problem.n() - 1;
      else
        MaxLocalKrons = 1;
    }

    uint UpperLocalKrons = problem.n();
    if (fastKron.cudaKernels.distComm_ == DistComm::NCCL && fastKron.cudaKernels.perGPUKronBatch_ == 1)
      UpperLocalKrons = 2;
    
    if (fastKron.cudaKernels.gpusInK_ == 1) {
      UpperLocalKrons = problem.n() + 1;
      MaxLocalKrons = problem.n();
    }

    //TODO: consider only valid krons 
    for (; MaxLocalKrons < UpperLocalKrons; MaxLocalKrons += 1) {
    float seriesTime = 0;
    TunedKernelsSeries tunedKernelSeries;

    auto tmpProblem = KMMProblem(problem.type(), Matrix(gpuM, gpuK,  temp1[0].data()), 
                                 problem.opX(), problem.n(), &Fs[0][0], problem.opFs(),
                                 Matrix(gpuM, problem.y().n()/fastKron.cudaKernels.gpusInK_, temp2[0].data()));

    for (int i = problem.n() - 1; i >= 0; i -= MaxLocalKrons) {
      const uint LocalKrons = std::min(MaxLocalKrons, i + 1);
      //TODO: any way to avoid declaring ps, qs, and fs on stack
      //set max value of N as 64
      auto subproblem = tmpProblem.rsub(i, LocalKrons);
      void* gpuResults[fastKron.cudaKernels.numGPUs_] = {nullptr};
      std::transform(temp2, temp2 + fastKron.cudaKernels.numGPUs_, &gpuResults[0], [](Matrix m) {return m.data();});
      DistributedParams distParams(0, 0, fastKron.cudaKernels.gpusInK_, subproblem.k() * fastKron.cudaKernels.gpusInK_, subproblem.l() * fastKron.cudaKernels.gpusInK_, 
                                   subproblem.k(), subproblem.l(), subproblem.fs(), 
                                   subproblem.n());
      distParams.updateGPUResults((void**)gpuResults);
      bool distP2PStore = fastKron.cudaKernels.gpusInK_ > 1 && fastKron.cudaKernels.isDistributed_ && fastKron.cudaKernels.distComm_ == DistComm::P2P;
      tune(subproblem, distP2PStore, distParams);
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
      fastKron.cudaKernels.perGPUKronBatch_ = MaxLocalKrons;
    }
    }
#endif
  }

  for (int p = 0; p < devicesPerProc; p++) {
    kernelDb->procFree(p, temp1[p]);
    kernelDb->procFree(p, temp2[p]);
    for (int f = 0; f < problem.n(); f++) {
      //TODO: // CUDA_CHECK(cudaFree(Fs[g * problem.n() + f]));
    }
  }
  
  std::cout <<"Minimum Time " << minTime << " through kernels: " << std::endl;
  for (auto iter = fastKron.tunedKernelSeries.rbegin(); iter != fastKron.tunedKernelSeries.rend(); iter++) {
    std::cout << "  " << (*iter) << std::endl;
#ifdef ENABLE_CUDA
    if (fastKron.cudaKernels.isDistributed_ and fastKron.cudaKernels.gpusInK_ > 1 and 
        ((problem.n() - iter->start) % fastKron.cudaKernels.perGPUKronBatch_ == 0 or 
        iter->start == 0)) {
      uint gpuM, gpuK;
      fastKron.getDistributedSizes(problem.m(), problem.k(), gpuM, gpuK);
      std::cout << "  " << "Communicate [" << gpuM << ", " << gpuK << "] among " << 
                   "[GM, " << fastKron.cudaKernels.gpusInK_ << "] using " << fastKron.cudaKernels.distComm_ << std::endl;
    }
#endif
  }
  return fastKronSuccess;
}