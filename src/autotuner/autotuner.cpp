#include <cassert>
#include <iostream>
#include <iomanip>

#include "autotuner/autotuner.h"
#include "handle/handle.h"
#include "kmm/kmmalgo.h"
#include "utils/utils.h"
#include "utils/logger.h"

/**
 * minExecTimeOfSeries() - Obtains execution time of fastest tuned kernel series for a problem.
 * @problem: The base KMMProblem.
 * @startF: Obtain fastest tuned kernel series of problem starting from this factor index.
 * @isDistributed: True if multiple GPUs are used or False.
 * @tunedKernels: [OUT] Output tuned kernel series
 * @tunedKernelsMap: Map of tuned kernels to KMMProblems
 * 
 * Return - Minimum execution time of the fastest tuned kernel series.
 *
 * A recursive combonitarial approach to go through every sub problem of the base problem and selects
 * a series of kernels that minimizes the total execution time of the base problem.
 */
template<typename KMMProblem, typename TunedKernelsMap>
static float minExecTimeOfSeries(KMMProblem problem, uint startF, bool isDistributed,
                                 TunedKernelsSeries& tunedKernels,
                                 TunedKernelsMap tunedKernelsMap) {
  if (startF >= problem.n()) return 0;
  float minTime = std::numeric_limits<float>::max();
  TunedKernelsSeries minEpilogueKernels;
  TunedKernelFromStart minPrologueKernel;

  //Obtain the subproblem starting at startF
  auto subProblem = problem.sub(startF, problem.n() - startF);

  //Divide the subproblem into two parts. Go through all first/second part pairs 
  //of the subproblem. Search for tuned kernel of the first part 
  //and recursively compute minimum time for the second part.
  reverseExecuteGeMKM(subProblem, nullptr, typename KMMProblem::Matrix(), 
                      [](const KMMProblem){return 1;},
    [&](const KMMProblem, int rstart, void*[2], typename KMMProblem::Matrix) {
      const int subn = rstart + 1;
      auto firstPart = problem.sub(startF, subn);
      if (problem.opX() == fastKronOp_T && startF + subn == problem.n()) {
        //If opX is T and the firstPart has reached end of the problem
        //then consider only TT or TN kernels
        firstPart.setOpX(fastKronOp_T);
      } else {
        firstPart.setOpX(fastKronOp_N);
      }
      //P2P is needed when the output of subproblem is distributed
      bool isP2P = isDistributed && startF == 0;
      if (tunedKernelsMap.hasKernel(firstPart, isP2P)) {
        //If the first part is tuned then recursively search for best kernel series for
        //the second part.
        TunedKernelsSeries epilogueKernels;
        float kernelTime = tunedKernelsMap.getKernelTime(firstPart, isP2P);
        float epilogueTime = minExecTimeOfSeries(problem, startF + rstart + 1,
                                                 isDistributed,
                                                 epilogueKernels, tunedKernelsMap);
        if (minTime > kernelTime + epilogueTime) {
          minTime = kernelTime + epilogueTime;
          minEpilogueKernels = epilogueKernels;
          minPrologueKernel = TunedKernelFromStart(tunedKernelsMap.getKernel(firstPart, isP2P),
                                                   startF, startF + rstart,
                                                   firstPart.k(), kernelTime);
        }
      }

      return fastKronSuccess;
    });

  //Combine tuned kernels for both parts
  tunedKernels = minEpilogueKernels;
  tunedKernels.push_back(minPrologueKernel);
  assert(minTime < std::numeric_limits<float>::max());
  return minTime;
}

/**
  * tune() - Tune kernels for all subproblems in the KMMProblem.
  * @problem: The base KMMProblem.
  * @kernelDb: KernelDatabase containing kernels.
  * @isDistributed: If the KMMProblem is computed using distributed GPUs
  * @distParams: Distributed paramaters if needed.
  */
template<typename KMMProblemT, typename TunedKernelsMap>
fastKronError Autotuner::tune(KMMProblemT problem, TunedKernelsMap& tunedKernelsMap,
                              KernelDatabase* kernelDb, bool isDistributed,
                              DistributedParams distParams) {
  //Iterate over all subproblems of the base problem
  auto err = reverseExecuteGeMKM(problem, nullptr, typename KMMProblemT::Matrix(), 
                                 [](const KMMProblemT){return 1;},
    [&](const KMMProblemT, int rstart, void*[2], typename KMMProblemT::Matrix) {
      for (uint32_t endP = rstart; endP < problem.n(); endP++) {
        //Obtain a subprob
        auto subprob = problem.sub(rstart, endP-rstart+1);
        //Only the first executed subprob has OpX as T otherwise
        //all subprob requires OpX as N
        if (rstart + subprob.n() < problem.n()) {
          subprob.setOpX(fastKronOp_N);
        }
        //P2P is needed when the output of subproblem is distributed
        bool p2p = isDistributed && rstart == 0;
        if (tunedKernelsMap.hasKernel(subprob, p2p) || 
            (!this->fastKron.getUseFusion() and subprob.n() > 1)) {
          //Avoid tuning if subprob is already tuned or
          //if fusion is disabled
          continue;
        }
        auto bestKernelWithTime = kernelDb->findTunedKernel(subprob, p2p, rstart, distParams);
        if (bestKernelWithTime.second < std::numeric_limits<float>::max()) {
          tunedKernelsMap.add(subprob, p2p, bestKernelWithTime);
        }
      }

      return fastKronSuccess;
    });

  return err;
}

/**
  * tune() - Find the best performing kernel series for a KMMProblem on a backend
  * @problem: KMMProblem
  * @backend: fastKronBackend containing kernels
  * @retKernelSeries: [OUT] the tuned kernel series 
  */
fastKronError Autotuner::tune(KMMProblem problem,
                              const fastKronBackend backend,
                              TunedKernelsSeries& retKernelSeries) {
  auto kernelDb = fastKron.getKernelDb(backend);
  //Return cached kernel series for the problem
  if (tunedProblemCache[kernelDb].count(problem) == 1) {
    retKernelSeries = tunedProblemCache[kernelDb][problem];
    return fastKronSuccess;
  }

  float minTime = 0;
  Matrix result, temp;
  fastKron.gekmmResultTemp(problem, result, temp);

  uint devicesPerProc = 1;

  if (fastKron.hasBackend(fastKronBackend_CUDA)) {
    #ifdef ENABLE_CUDA
      devicesPerProc = ((CUDAKernelDatabase*)fastKron.getKernelDb(fastKronBackend_CUDA))->numGPUs_;
    #endif
  } else if (fastKron.hasBackend(fastKronBackend_X86)) {
      devicesPerProc = 1;
  }

  Matrix temp1[devicesPerProc];
  Matrix temp2[devicesPerProc];
  //TODO: Make this FactorArray
  Factor Fs[devicesPerProc][problem.n()];
  for (uint32_t p = 0; p < devicesPerProc; p++) {
    //For performance eval we do not need these to contain any specific value
    fastKron.gekmmResultTemp(problem, result, temp1[p]);
    fastKron.gekmmResultTemp(problem, result, temp2[p]);
    kernelDb->procMalloc(p, problem.type(), temp1[p]);
    kernelDb->procMalloc(p, problem.type(), temp2[p]);
    kernelDb->procMemset(p, temp1[p], 1.0f);
    kernelDb->procMemset(p, temp2[p], 1.0f);

    for (uint32_t f = 0; f < problem.n(); f++) {
      Fs[p][f] = problem.f(f);
      kernelDb->procMalloc(p, problem.type(), Fs[p][f]);
      kernelDb->procMemset(p, Fs[p][f], 1.0f);
    }
  }

  kernelDb->initTune();

  if (devicesPerProc <= 1) {
    //Tuning for Single CPU / Single GPU
    //Use temporary as input/output matrix
    Matrix x(problem.x().m(), problem.x().n(), temp1[0].data());
    Matrix y(problem.y().m(), problem.y().n(), temp2[0].data());

    KMMProblem tmpProblem(FastKronMMType::MKM, problem.type(), x, problem.opX(), 
                          problem.n(), &Fs[0][0], problem.opFs(), y);
    tune(tmpProblem, tunedKernelsMap, kernelDb, false, DistributedParams());
    Logger(LogLevel::Debug) << "Finding min execution time of the series" << std::endl;
    minTime = minExecTimeOfSeries(problem, 0, false, retKernelSeries, tunedKernelsMap);
  } else {
#if defined(ENABLE_CUDA) && defined(ENABLE_MULTI_GPU)
    //Tuning for Multi GPU
    //TODO: Document this code
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

  for (uint32_t p = 0; p < devicesPerProc; p++) {
    kernelDb->procFree(p, temp1[p]);
    kernelDb->procFree(p, temp2[p]);
    for (uint32_t f = 0; f < problem.n(); f++) {
      kernelDb->procFree(p, Fs[p][f]);
    }
  }
  
  Logger(LogLevel::Info) << "Minimum Time " << minTime << " through kernels: " << std::endl;
  for (auto iter = retKernelSeries.rbegin(); iter != retKernelSeries.rend(); iter++) {
    Logger(LogLevel::Info) << "  " << (*iter) << std::endl;
#if defined(ENABLE_CUDA) && defined(ENABLE_MULTI_GPU)
    if (fastKron.cudaKernels.isDistributed_ and fastKron.cudaKernels.gpusInK_ > 1 and 
        ((problem.n() - iter->start) % fastKron.cudaKernels.perGPUKronBatch_ == 0 or 
        iter->start == 0)) {
      uint gpuM, gpuK;
      fastKron.getDistributedSizes(problem.m(), problem.k(), gpuM, gpuK);
      Logger(LogLevel::Info) << "  " << "Communicate [" << gpuM << ", " << gpuK << "] among " << 
                   "[GM, " << fastKron.cudaKernels.gpusInK_ << "] using " << fastKron.cudaKernels.distComm_ << std::endl;
    }
#endif
  }

  //Update cache
  tunedProblemCache[kernelDb][problem] = retKernelSeries;

  return fastKronSuccess;
}
  
fastKronError Autotuner::tune(KMMProblemStridedBatched problem, const fastKronBackend backend,
                              TunedKernelsSeries& retKernelSeries) {
  auto kernelDb = fastKron.getKernelDb(backend);
  //Return cached kernel series for the problem
  if (tunedProblemCacheStridedBatched[kernelDb].count(problem) == 1) {
    retKernelSeries = tunedProblemCacheStridedBatched[kernelDb][problem];
    return fastKronSuccess;
  }

  uint devicesPerProc = 1;


  if (devicesPerProc > 1) {
    Logger(LogLevel::Debug) << "StridedBatched is not supported for multi GPU" << std::endl;
    abort();
  }

  float minTime = 0;
  KMMProblemStridedBatched::Matrix result, temp;
  fastKron.gekmmResultTemp(problem, result, temp);

  KMMProblemStridedBatched::Matrix temp1[devicesPerProc];
  KMMProblemStridedBatched::Matrix temp2[devicesPerProc];

  //TODO: Make this FactorArray
  KMMProblemStridedBatched::Factor Fs[devicesPerProc][problem.n()];
  for (uint32_t p = 0; p < devicesPerProc; p++) {
    //For performance eval we do not need these to contain any specific value
    fastKron.gekmmResultTemp(problem, result, temp1[p]);
    fastKron.gekmmResultTemp(problem, result, temp2[p]);
    kernelDb->procMalloc(p, problem.type(), temp1[p], problem.batchCount());
    kernelDb->procMalloc(p, problem.type(), temp2[p], problem.batchCount());
    kernelDb->procMemset(p, temp1[p], problem.batchCount(), 1.0f);
    kernelDb->procMemset(p, temp2[p], problem.batchCount(), 1.0f);

    for (uint32_t f = 0; f < problem.n(); f++) {
      Fs[p][f] = problem.f(f);
      kernelDb->procMalloc(p, problem.type(), Fs[p][f], problem.batchCount());
      kernelDb->procMemset(p, Fs[p][f], problem.batchCount(), 1.0f);
    }
  }

  kernelDb->initTune();

  if (devicesPerProc <= 1) {
    //Tuning for Single CPU / Single GPU
    //Use temporary as input/output matrix
    KMMProblemStridedBatched::Matrix x = problem.x().like(temp1[0].data());
    KMMProblemStridedBatched::Matrix y = problem.y().like(temp2[0].data());

    KMMProblemStridedBatched tmpProblem(FastKronMMType::MKM, problem.type(), x, problem.opX(),
                                        problem.n(), &Fs[0][0], problem.opFs(), y,
                                        problem.batchCount());
    tune(tmpProblem, tunedKernelsMapStridedBatched, kernelDb, false, DistributedParams());
    Logger(LogLevel::Debug) << "Finding min execution time of the series" << std::endl;
    minTime = minExecTimeOfSeries(problem, 0, false, retKernelSeries, tunedKernelsMapStridedBatched);
    Logger(LogLevel::Info) << "Minimum Time " << minTime << " through kernels: " << std::endl;
    for (auto iter = retKernelSeries.rbegin(); iter != retKernelSeries.rend(); iter++) {
      Logger(LogLevel::Info) << "  " << (*iter) << std::endl;
#if defined(ENABLE_CUDA) && defined(ENABLE_MULTI_GPU)
      if (fastKron.cudaKernels.isDistributed_ and fastKron.cudaKernels.gpusInK_ > 1 and 
          ((problem.n() - iter->start) % fastKron.cudaKernels.perGPUKronBatch_ == 0 or 
          iter->start == 0)) {
        uint gpuM, gpuK;
        fastKron.getDistributedSizes(problem.m(), problem.k(), gpuM, gpuK);
        Logger(LogLevel::Info) << "  " << "Communicate [" << gpuM << ", " << gpuK << "] among " << 
                    "[GM, " << fastKron.cudaKernels.gpusInK_ << "] using " << fastKron.cudaKernels.distComm_ << std::endl;
      }
#endif
    }
  }

  //Update cache
  tunedProblemCacheStridedBatched[kernelDb][problem] = retKernelSeries;

  return fastKronSuccess;
}

Autotuner::Autotuner(FastKronHandle& fastKron) : fastKron(fastKron) {
  for (auto db : fastKron.getAllKernelDbs()) {
    tunedProblemCache[db] = std::unordered_map<KMMProblem, TunedKernelsSeries>();
  }
}