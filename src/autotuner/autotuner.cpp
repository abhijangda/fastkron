#include <cassert>
#include <iostream>
#include <iomanip>

#include "autotuner/autotuner.h"
#include "handle/handle.h"
#include "kmm/kmmalgo.h"
#include "utils/utils.h"
#include "utils/debug_print.h"

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

fastKronError Autotuner::tune(KMMProblem problem, KernelDatabase* kernelDb,
                            bool isDistributed, DistributedParams distParams) {
  //Only row major layout of all matrics is supported.
  //For performance eval we do not need these to contain any value
  
  //Use double buffering for writing result and using output 
  //of previous iteration as input to current
  //A KronMat is a series of SlicedMats
  //We need to get best kernel for all contiguous SlicedMats

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

fastKronError Autotuner::tune(KMMProblem problem, const fastKronBackend backend, TunedKernelsSeries& retKernelSeries) {
  //Only row major layout of all matrics is supported.
  auto kernelDb = fastKron.getKernelDb(backend);

  if (tunedKernelSeries[kernelDb].count(problem) == 1) {
    retKernelSeries = tunedKernelSeries[kernelDb][problem];
    return fastKronSuccess;
  }

  float minTime = 0;
  Matrix result, temp;
  fastKron.gekmmResultTemp(problem, result, temp);

  uint devicesPerProc = 1;

  if (fastKron.hasBackend(fastKronBackend_CUDA)) {
    #ifdef ENABLE_CUDA
      devicesPerProc = fastKron.cudaKernels.numGPUs_;
    #endif
  } else if (fastKron.hasBackend(fastKronBackend_X86)) {
      devicesPerProc = 1;
  }

  Matrix temp1[devicesPerProc];
  Matrix temp2[devicesPerProc];
  //TODO: Make this FactorArray
  Factor Fs[devicesPerProc][problem.n()];
  for (uint32_t p = 0; p < devicesPerProc; p++) {
    //TODO: Init temp to 1
    fastKron.gekmmResultTemp(problem, result, temp1[p]);
    fastKron.gekmmResultTemp(problem, result, temp2[p]);
    kernelDb->procMalloc(p, problem.type(), temp1[p]);
    kernelDb->procMalloc(p, problem.type(), temp2[p]);
    kernelDb->procMemset(p, temp1[p], 1.0f);
    kernelDb->procMemset(p, temp2[p], 1.0f);

    for (int f = 0; f < problem.n(); f++) {
      Fs[p][f] = problem.f(f);
      kernelDb->procMalloc(p, problem.type(), Fs[p][f]);
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
    tune(tmpProblem, kernelDb, false, DistributedParams());
    DebugPrint(LogLevel::Debug) << "Finding min execution time of the series" << std::endl;
    TunedKernelsSeries tunedKernels;
    minTime = minExecTimeOfSeries(problem, 0, false,
                                  tunedKernels, tunedKernelsMap);
    retKernelSeries = tunedKernels;
  } else {
#ifdef ENABLE_CUDA
    assert(fastKron.hasBackend(fastKronBackend_CUDA));
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
      tune(subproblem, kernelDb, distP2PStore, distParams);
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
      retKernelSeries = tunedKernelSeries;
      distribTunedKernelSeries = tunedKernelSeries;
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
  
  DebugPrint(LogLevel::Info) << "Minimum Time " << minTime << " through kernels: " << std::endl;
  for (auto iter = retKernelSeries.rbegin(); iter != retKernelSeries.rend(); iter++) {
    DebugPrint(LogLevel::Info) << "  " << (*iter) << std::endl;
#ifdef ENABLE_CUDA
    if (fastKron.cudaKernels.isDistributed_ and fastKron.cudaKernels.gpusInK_ > 1 and 
        ((problem.n() - iter->start) % fastKron.cudaKernels.perGPUKronBatch_ == 0 or 
        iter->start == 0)) {
      uint gpuM, gpuK;
      fastKron.getDistributedSizes(problem.m(), problem.k(), gpuM, gpuK);
      DebugPrint(LogLevel::Info) << "  " << "Communicate [" << gpuM << ", " << gpuK << "] among " << 
                   "[GM, " << fastKron.cudaKernels.gpusInK_ << "] using " << fastKron.cudaKernels.distComm_ << std::endl;
    }
#endif
  }

  tunedKernelSeries[kernelDb][problem] = retKernelSeries;

  return fastKronSuccess;
}

Autotuner::Autotuner(FastKronHandle& fastKron) : fastKron(fastKron) {
  for (auto db : fastKron.getAllKernelDbs()) {
    tunedKernelSeries[db] = std::unordered_map<KMMProblem, TunedKernelsSeries>();
  }
}