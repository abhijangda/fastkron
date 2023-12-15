#include "handle/handle.h"
#include "kmm/kmmalgo.h"

#pragma once

class TunedKernelsMap {
  using ProblemToKernels = std::unordered_map<KMMProblem, std::pair<KernelInfo, float>>;

  ProblemToKernels kernels;
  ProblemToKernels p2pKernels;

  ProblemToKernels::const_iterator getKernel(const ProblemToKernels& map, const KMMProblem& problem) {
    return map.find(problem);
  }

public:
  TunedKernelsMap() {}

  void add(const KMMProblem& problem, bool p2p, std::pair<KernelInfo, float> kernelAndtime) {
    if (p2p) {
      p2pKernels.emplace(std::make_pair(problem, kernelAndtime));
    } else {
      kernels.emplace(std::make_pair(problem, kernelAndtime));
    }
  }

  bool hasKernel(const KMMProblem& problem, bool p2p) {
    return (p2p) ? getKernel(p2pKernels, problem) != p2pKernels.end():
                   getKernel(kernels, problem) != kernels.end();
  }

  KernelInfo getKernel(const KMMProblem& problem, bool p2p) {
    return (p2p) ? getKernel(p2pKernels, problem)->second.first :
                   getKernel(kernels, problem)->second.first;    
  }

  float getKernelTime(const KMMProblem& problem, bool p2p) {
    return (p2p) ? getKernel(p2pKernels, problem)->second.second :
                   getKernel(kernels, problem)->second.second;    
  }
};

class Autotuner {
  FastKronHandle& fastKron;
  TunedKernelsMap tunedKernelsMap;

  cudaError_t tuneSlicedMulSeries(KMMProblem problem,
                                  bool isDistributed, DistributedParams distParams,
                                  cudaStream_t stream);
public:
  Autotuner(FastKronHandle& fastKron) : fastKron(fastKron)
  {}

  cudaError_t tune(KMMProblem problem, cudaStream_t stream);

};