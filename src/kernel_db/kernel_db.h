#include <functional>
#include <map>
#include <algorithm>

#include "kmm/kmmalgo.h"
#include "kernels/kmmkernel.h"
#include "kernels/params.h"
#include "utils/logger.h"

#pragma once

/**
 * A KernelDatabase contains a database of all compiled kernels for a backend.
 * Each backend has a subclass of KernelDatabase.
 */
class KernelDatabase {
public:
  /**
   * DbKey is key for to map (Factor, fastKronOp for X, and fastKronOp for F) to kernel.
   */
  struct DbKey {
    Factor f;
    fastKronOp opX;
    fastKronOp opF;
    KernelBatchType::Ty batchType;
 
    bool operator==(const DbKey& other) const {
      return other.f == f && other.opX == opX && other.opF == opF && other.batchType == batchType;
    }
  };

  /**
   * DbKeyHash is a functor to obtain hash for DbKey.
   */
  struct DbKeyHash {
    size_t operator()(const DbKey& k) const {
      return std::hash<Factor>  ()(k.f)        ^
             std::hash<uint32_t>()(k.opX)      ^
             std::hash<uint32_t>()(k.opF)      ^
             std::hash<uint32_t>()(k.batchType);
    }
  };
  
  /**
   * @compiledKernels: A map of DbKey, i.e., (Factor, fastKronOp for X, and fastKronOp for F) to kernels 
   *                   that can process this Factor, OpX, and OpF.
   */
  std::unordered_map<DbKey, std::vector<KMMKernel*>, DbKeyHash> compiledKernels;

  /**
   * @hardware: A vector of all underlying hardware (CPUs or GPUs) for this backend.
   */
  std::vector<HardwareDetails*> hardware;
  
  /**
   * @problemToKernelCache: A map of KMMProblem to already tuned kernels for this backend.
   */
  std::unordered_map<KMMProblem, TunedKernelsSeries> problemToKernelCache;
  std::unordered_map<KMMProblemStridedBatched, TunedKernelsSeries> stridedBatchedProblemToKernelCache;

public:
  KernelDatabase();
  ~KernelDatabase() {}

protected:
  /**
   * loadKernels() - Process all kernels and add kernels to the database.
   * @SubClassKernel: Type of kernel for backend's subclass. 
   * @kernels: Array of kernels.
   * @num: Number of elements of the array.
   */
  template<typename SubClassKernel> void loadKernels(SubClassKernel* kernels, uint32_t num);

  /*********************** Memory Allocation Methods ***********************/
public:
  /**
   * procMalloc() - Allocate buffer for matrix on a given process.
   * @proc: Process number, e.g., GPU number in a multi-GPU case.
   * @type: Datatype.
   * @m: [OUT] Output Matrix.
   * 
   * Return: fastKronSuccess if no error otherwise a fastKronError.
   */
  fastKronError procMalloc(uint32_t proc, FastKronType type, Matrix& m);
  fastKronError procMalloc(uint32_t proc, FastKronType type, StridedBatchMatrix& m, int batches);
  fastKronError procMalloc(uint32_t proc, FastKronType type, StridedBatchFactor& m, int batches);

  /**
   * procFree() - Frees buffer for matrix on a given process.
   * @proc: Process number, e.g., GPU number in a multi-GPU case.
   * @m: Matrix containing buffer.
   *
   * Return: fastKronSuccess if no error otherwise a fastKronError.
   */
  fastKronError procFree(uint32_t proc, Matrix m);

  /**
   * procMemset() - Set same value to each element of the Matrix on a given process.
   * @proc: Process number, e.g., GPU number in a multi-GPU case.
   * @m: [OUT] Matrix
   * @val: float value to set
   *
   * Return: fastKronSuccess if no error otherwise a fastKronError.
   */
  virtual fastKronError procMemset(uint32_t proc, Matrix& m, float val) = 0;

  fastKronError procMemset(uint32_t proc, StridedBatchMatrix& m, int batches, float val);
  fastKronError procMemset(uint32_t proc, StridedBatchFactor& m, int batches, float val);

protected:
  /**
   * procMalloc(), procFree() - Allocate/free buffer on a given process.
   * @proc: Process number, e.g., GPU number in a multi-GPU case.
   * @size: Size of buffer in bytes to allocate.
   * @ptr: [OUT] allocated/freed pointer
   *
   * Return: fastKronSuccess if no error otherwise a fastKronError.
   */
  virtual fastKronError procMalloc(uint32_t proc, size_t size, void*& ptr) = 0;
  virtual fastKronError procFree(uint32_t proc, void* ptr) = 0;
    
  /***************************************************************************/

  /*********************** Kernel Invocation Methods ***********************/
public:
  /**
   * invokeKernel() - Invokes a kernel to compute GeKMM for a factor. 
   *                  This method must be defined by each KernelDatabase.
   * @kernel: kernel to invoke.
   * @problem: KMMProblem to compute.
   * @fidx: Factor index in the KMMProblem.
   * @eplogueParams: Parameter for Epilogue (alpha, beta, and Y)
   * @execMode: Execution mode
   */
  virtual fastKronError invokeKernel(KMMKernel* kernel, KMMProblem problem,
                                     const uint fidx,
                                     EpilogueParams epilogueParams,
                                     KernelMode execMode) = 0;
  virtual fastKronError invokeKernel(KMMKernel* kernel, KMMProblemStridedBatched problem,
                                     const uint fidx,
                                     EpilogueStridedBatchedParams epilogueParams,
                                     KernelMode execMode) = 0;
  /**
   * invokeP2PStoreKernel()- Invokes a P2P kernel to compute GeKMM for a factor 
   *                         and write output among all nodes using RDMA.
   *                         This method must be defined by each KernelDatabase.
   * @kernel: kernel to invoke.
   * @problem: KMMProblem to compute.
   * @fidx: Factor index in the KMMProblem.
   * @distParams: Parameters for Distributed 
   * @eplogueParams: Parameter for Epilogue (alpha, beta, and Y)
   * @execMode: Execution mode
   */
  virtual fastKronError invokeP2PStoreKernel(KMMKernel* kernel, KMMProblem problem,
                                             const uint fidx,  
                                             DistributedParams distParams, 
                                             EpilogueParams epilogueParams,
                                             KernelMode execMode) = 0;
  /***************************************************************************/

  /************************* Auto tuning Methods ***************************/

public:
  /**
   * initTune() - Initialize auto tuning. This method is called by autotuner before starting
   *              the auto tuning process. This method must be defined by every KernelDatabase.
   */
  virtual fastKronError initTune() = 0;

  /**
   * timeKernel() - Obtain execution time of a kernel for computing GeKMM for a factor.
   * @kernel: kernel to invoke.
   * @problem: KMMProblem to compute.
   * @fidx: Factor index in the KMMProblem.
   * @distParams: Parameters for Distributed.
   * @eplogueParams: Parameter for Epilogue (alpha, beta, and Y).
   * @execMode: Execution mode.
   * @useP2PStore: True if uses RDMA P2P store otherwise false.
   * @warmups: Number of warming up runs.
   * @runs: Number of times to run kernel.
   * @runtime: [OUT] Average time of @runs after @warmups.
   */
  virtual fastKronError timeKernel(KMMKernel* kernel, KMMProblem problem, 
                                   const uint fidx, 
                                   DistributedParams distParams,
                                   EpilogueParams epilogueParams,
                                   KernelMode execMode, 
                                   bool useP2PStore,
                                   int warmups, int runs,
                                   float& runtime) = 0;

  virtual fastKronError timeKernel(KMMKernel* kernel, KMMProblemStridedBatched problem, 
                                   const uint fidx, 
                                   DistributedParams distParams,
                                   EpilogueStridedBatchedParams epilogueParams,
                                   KernelMode execMode, 
                                   bool useP2PStore,
                                   int warmups, int runs,
                                   float& runtime) = 0;
  /**
   * tuneKernelForProblem() - Find tuned kernel for problem.
   * @problem: Find tuned kernel for computing this problem.
   * @useP2PStore: True if computing this problem requires P2P RDMA store.
   * @fidx: Index of factor in the parent problem.
   * @distParams: Distributed parameters.
   *
   * Return - A pair of tuned kernel and execution time of this kernel.
   */
  std::pair<KMMKernel*, float> findTunedKernel(KMMProblem subproblem, 
                                                bool useP2PStore, uint fidx,
                                                DistributedParams distParams);
  std::pair<KMMKernel*, float> findTunedKernel(KMMProblemStridedBatched subproblem, 
                                                bool useP2PStore, uint fidx,
                                                DistributedParams distParams);
  /***************************************************************************/

  /*********************** Kernel Search Methods ***************************/
public:
  /**
   * kernelSeriesForProblem() - Top level method to get kernel series for a problem using FastKron's 
   *                            kernel search algorithm.
   * @problem: The problem to search kernel series for.
   *
   * Return - An object of TunedKernelSeries.
   */
  TunedKernelsSeries kernelSeriesForProblem(KMMProblem problem);
  TunedKernelsSeries kernelSeriesForProblem(KMMProblemStridedBatched problem);

private:
  template<typename KMMProblemT, typename KernelCache>
  TunedKernelsSeries kernelSeriesForProblem(KMMProblemT problem, KernelCache problemToKernelCache);

  /**
   * tuneKernelForProblem() - Find tuned kernel for problem.
   * @problem: Find tuned kernel for computing this problem.
   * @useP2PStore: True if computing this problem requires P2P RDMA store.
   * @fidx: Index of factor in the parent problem.
   * @distParams: Distributed parameters.
   *
   * Return - A pair of tuned kernel and execution time of this kernel.
   */
  template<typename KMMProblemT, typename EpilogueParams>
  std::pair<KMMKernel*, float> findTunedKernel(KMMProblemT subproblem, KernelBatchType::Ty batchType,
                                                bool useP2PStore, uint fidx,
                                                DistributedParams distParams);
  /**
   * findAllFusedKernels() - Find all fused kernels that can compute given problem.
   * @problem: The problem to find kernels for.
   * @useP2PStore: True if kernels should use P2P RDMA stores.
   * @kernels: [OUT] A vector of found fused kernels.
   *
   * Return - True if atleast one kernel is found, otherwise false.
   */
  bool findAllFusedKernels(KMMProblem problem, bool useP2PStore, 
                           std::vector<KMMKernel*>& kernels,
                           KernelBatchType::Ty batchType = KernelBatchType::Normal);
  bool findAllFusedKernels(KMMProblemStridedBatched problem, bool useP2PStore, std::vector<KMMKernel*>& kernels);
  
  /**
   * findAllKernels() - Find all kernels (fused or non-fused) that can compute the given problem.
   * @problem: The problem to find kernels for.
   * @useP2PStore: True if kernels should use P2P RDMA stores.
   * @kernels: [OUT] A vector of kernels for each optimization level.
   *
   * Return - True if atleast one kernel is found, otherwise false. 
   */
  template<typename KMMProblem>
  bool findAllKernels(KMMProblem problem, KernelBatchType::Ty batchType, bool useP2PStore,
                      std::vector<std::vector<KMMKernel*>>& kernels);

protected:
  /**
   * findKernelForSubProblem() - Find best kernel for a sub problem 
   *                             (of single factor) from a map of opt level to kernels. 
   * @subProblem: A sub problem with n() == 1
   * @kernels: A vector of OptLevel to a vector of kernels at that level.
   * 
   * Return - The best kernel for the sub problem using FastKron's kernel search algorithm.
   */
  template<typename KMMProblemT>
  KMMKernel* findKernelForSubProblem(KMMProblemT subProblem, const std::vector<std::vector<KMMKernel*>>& kernels);

  /**
   * findKernelAtOptLevel() - Find best kernel for a subproblem at an opt level. This method should be 
   *                          implemented by each Kernel Database.
   * @subproblem: A sub problem with n() == 1.
   * @kernelsForOptLevel: A vector of kernels at an opt level.
   *
   * Return - The best kernel found for the opt level.
   */
  virtual KMMKernel* findKernelAtOptLevel(KMMProblem subProblem, const std::vector<KMMKernel*>& kernelsForOptLevel) = 0;
  virtual KMMKernel* findKernelAtOptLevel(KMMProblemStridedBatched subProblem, const std::vector<KMMKernel*>& kernelsForOptLevel) = 0;

  /**
   * getMapNumFusedProbsToKernels() - Filter all fused kernels to find fastest fused kernels for a problem.
   * @problem: The problem to find kernels for.
   * @kernels: The set of kernels to filter from.
   * 
   * Return - A map of number of fused problems -> a vector of all fused kernels for the number.
   */
  virtual std::map<uint32_t, std::vector<KMMKernel*>, std::greater<int>> 
          filterFastestFusedKernels(const KMMProblem& problem, const std::vector<KMMKernel*>& kernels);
  std::map<uint32_t, std::vector<KMMKernel*>, std::greater<int>> 
          filterFastestFusedKernels(const KMMProblemStridedBatched& problem, const std::vector<KMMKernel*>& kernels) {
    return filterFastestFusedKernels(problem.batchProblem(0), kernels);
  }
  /***************************************************************************/

  /**************************** Helper Methods *****************************/
protected:
  /**
   * occupancyDetails() - A pure virtual method to obtain Kernel Occupancy detail as a string. 
   *                      This method should be implemented by every KernelDatabase.
   * @kernel: The kernel to get occupancy details.
   * @problem: The problem computed by the kernel.
   *
   * Return: A string of occupancy details.
   */
  virtual std::string occupancyDetails(KMMKernel* kernelInfo, KMMProblem problem) = 0;
  virtual std::string occupancyDetails(KMMKernel* kernelInfo, KMMProblemStridedBatched problem) = 0;

public:
  /**
   * getKernel() - Obtain a kernel with the given string representation.
   * @repr: string representation.
   *
   * Return: The kernel found or nullptr if no kernel found.
   */
  KMMKernel* getKernel(std::string repr);
  /***************************************************************************/
};

#include "kernel_db/kernel_db.inline.h"