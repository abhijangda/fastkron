#include "kmm/kmmalgo.h"

#include "kernel_db/kernel_db.h"
#include "kernels/kmmkernel.h"

#include <utility>
#include <unordered_map>

#pragma once

/**
 * TunedKernelsMap - maps a KMMProblem to tuned kernel with its execution time.
 */
class TunedKernelsMap {
  /**
   * @ProblemToKernels: a map of KMMProblem to a pair of kernel and its run time in milliseconds.
   */
  using ProblemToKernels = std::unordered_map<KMMProblem, std::pair<KMMKernel*, float>>;

  /**
   * @kernels: the map of KMMProblem to single gpu/cpu kernels.
   * @p2pKernels: the map of KMMProblem to kernels storing output using P2P.
   */
  ProblemToKernels kernels;
  ProblemToKernels p2pKernels;

  /**
   * getKernel - get kernel of a problem from a map.
   */
  ProblemToKernels::const_iterator getKernel(const ProblemToKernels& map, const KMMProblem& problem) {
    return map.find(problem);
  }

public:
  TunedKernelsMap() {}

  /**
   * add() - add or update a kernel-time pair for a problem.
   * @problem: The KMMProblem to add kernel-time pair for.
   * @p2p: True if the problem requires P2P stores for storing output otherwise false.
   * @kernelAndTime: The pair of kernel and its runtime.
   */
  void add(const KMMProblem& problem, bool p2p, std::pair<KMMKernel*, float> kernelAndtime) {
    if (p2p) {
      p2pKernels.emplace(std::make_pair(problem, kernelAndtime));
    } else {
      kernels.emplace(std::make_pair(problem, kernelAndtime));
    }
  }

  /**
   * hasKernel() - Determine if there is a kernel for a problem
   * @problem: The KMMProblem to find kernel-time pair for 
   * @p2p: True if the problem requires P2P stores for storing output otherwise false.
   */
  bool hasKernel(const KMMProblem& problem, bool p2p) {
    return (p2p) ? getKernel(p2pKernels, problem) != p2pKernels.end():
                   getKernel(kernels,    problem) != kernels.end();
  }

  /**
   * getKernel() - Get the kernel for a problem
   * @problem: The KMMProblem to find kernel for 
   * @p2p: True if the problem requires P2P stores for storing output otherwise false.
   */
  KMMKernel* getKernel(const KMMProblem& problem, bool p2p) {
    return (p2p) ? getKernel(p2pKernels, problem)->second.first :
                   getKernel(kernels,    problem)->second.first;    
  }

  /**
   * getKernelTime() - Get the kernel-time for a problem
   * @problem: The KMMProblem to find kernel for 
   * @p2p: True if the problem requires P2P stores for storing output otherwise false.
   */
  float getKernelTime(const KMMProblem& problem, bool p2p) {
    return (p2p) ? getKernel(p2pKernels, problem)->second.second :
                   getKernel(kernels,    problem)->second.second;    
  }
};

/**
 * Forward declaration of FastKronHandle for Autotuner
 */
class FastKronHandle;

/**
 * Autotuner - goes through all valid kernels and finds a kernel series with least execution time 
 * for a KMMProblem
 */
class Autotuner {
  /**
   * @fastKron: The parent FastKron handle
   */
  FastKronHandle& fastKron;

  /**
   * @tunedKernelsMap: A map of tuned kernels and KMMProblems
   */
  TunedKernelsMap tunedKernelsMap;

  /**
   * @tunedProblemCache: A cache of already tuned full KMMProblems.
   * Maps each KMMProblem to tuned kernel series for each backend.
   */
  std::unordered_map<KernelDatabase*, std::unordered_map<KMMProblem, TunedKernelsSeries>> tunedProblemCache;

  /**
   * tune() - Tune kernels for all subproblems in the KMMProblem.
   * @problem: The base KMMProblem.
   * @kernelDb: KernelDatabase containing kernels.
   * @isDistributed: If the KMMProblem is computed using distributed GPUs
   * @distParams: Distributed paramaters if needed.
   */
  fastKronError tune(KMMProblem problem, KernelDatabase* kernelDb,
                     bool isDistributed, DistributedParams distParams);

public:
  TunedKernelsSeries distribTunedKernelSeries;

  Autotuner(FastKronHandle& fastKron);

  /**
   * tune() - Find the best performing kernel series for a KMMProblem on a backend
   * @problem: KMMProblem
   * @backend: fastKronBackend containing kernels
   * @retKernelSeries: [OUT] the tuned kernel series 
   */
  fastKronError tune(KMMProblem problem, const fastKronBackend backend, 
                     TunedKernelsSeries& retKernelSeries);
};