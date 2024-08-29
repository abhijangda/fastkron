#include <functional>
#include <map>
#include <algorithm>

#include "kmm/kmmalgo.h"
#include "kernels/kernel_info.h"
#include "kernels/params.h"
#include "utils/logger.h"

#pragma once

/**
 * A KernelDatabase contains a database of all compiled kernels for a backend.
 * Each backend has a subclass of KernelDatabase.
 */
class KernelDatabase {
protected:
  /**
   * DbKey is key for to map (Factor, fastKronOp for X, and fastKronOp for F) to kernel.
   */
  struct DbKey {
    Factor f;
    fastKronOp opX;
    fastKronOp opF;

    bool operator==(const DbKey& other) const {
      return other.f == f && other.opX == opX && other.opF == opF;
    }
  };

  /**
   * DbKeyHash is a functor to obtain hash for DbKey.
   */
  struct DbKeyHash {
    size_t operator()(const DbKey& k) const {
      return std::hash<Factor>  ()(k.f)   ^
             std::hash<uint32_t>()(k.opX) ^
             std::hash<uint32_t>()(k.opF);
    }
  };
  
  /**
   * @compiledKernels: A map of DbKey, i.e., (Factor, fastKronOp for X, and fastKronOp for F) to kernels 
   *                   that can process this Factor, OpX, and OpF.
   */
  std::unordered_map<DbKey, std::vector<KernelInfo*>, DbKeyHash> compiledKernels;

  /**
   * @hardware: A vector of all underlying hardware (CPUs or GPUs) for this backend.
   */
  std::vector<HardwareDetails*> hardware;
  
  /**
   * @problemToKernelCache: A map of KMMProblem to already tuned kernels for this backend.
   */
  std::unordered_map<KMMProblem, TunedKernelsSeries> problemToKernelCache;

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

  /*********************** Memory Allocation Functions ***********************/
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

  /*********************** Kernel Invocation Functions ***********************/
public:
  /**
   * invokeKernel() - Invokes a kernel to compute GeKMM for a factor. 
   *                  This function must be defined by each KernelDatabase.
   * @kernel: kernel to invoke.
   * @problem: KMMProblem to compute.
   * @fidx: Factor index in the KMMProblem.
   * @eplogueParams: Parameter for Epilogue (alpha, beta, and Y)
   * @execMode: Execution mode
   */
  virtual fastKronError invokeKernel(KernelInfo* kernel, KMMProblem problem,
                                     const uint fidx,
                                     EpilogueParams epilogueParams,
                                     KernelMode execMode) = 0;

  /**
   * invokeP2PStoreKernel()- Invokes a P2P kernel to compute GeKMM for a factor 
   *                         and write output among all nodes using RDMA.
   *                         This function must be defined by each KernelDatabase.
   * @kernel: kernel to invoke.
   * @problem: KMMProblem to compute.
   * @fidx: Factor index in the KMMProblem.
   * @distParams: Parameters for Distributed 
   * @eplogueParams: Parameter for Epilogue (alpha, beta, and Y)
   * @execMode: Execution mode
   */
  virtual fastKronError invokeP2PStoreKernel(KernelInfo* kernel, KMMProblem problem,
                                             const uint fidx,  
                                             DistributedParams distParams, 
                                             EpilogueParams epilogueParams,
                                             KernelMode execMode) = 0;
  /***************************************************************************/

  /************************* Auto tuning Functions ***************************/

public:
  /**
   * initTune() - Initialize auto tuning. This function is called by autotuner before starting
   *              the auto tuning process. This function must be defined by every KernelDatabase.
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
  virtual fastKronError timeKernel(KernelInfo* kernel, KMMProblem problem, 
                                   const uint fidx, 
                                   DistributedParams distParams,
                                   EpilogueParams epilogueParams,
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
  std::pair<KernelInfo*, float> findTunedKernel(KMMProblem subproblem, 
                                                bool useP2PStore, uint fidx,
                                                DistributedParams distParams);

  /***************************************************************************/

  /*********************** Kernel Search Functions ***************************/
public:
  /**
   * kernelSeriesForProblem() - Top level function to get kernel series for a problem using FastKron's 
   *                            kernel search algorithm.
   * @problem: The problem to search kernel series for.
   *
   * Return - An object of TunedKernelSeries.
   */
  TunedKernelsSeries kernelSeriesForProblem(KMMProblem problem);

private:
  /**
   * findAllFusedKernels() - Find all fused kernels that can compute given problem.
   * @problem: The problem to find kernels for.
   * @useP2PStore: True if kernels should use P2P RDMA stores.
   * @kernels: [OUT] A vector of found fused kernels.
   *
   * Return - True if atleast one kernel is found, otherwise false.
   */
  bool findAllFusedKernels(KMMProblem problem, bool useP2PStore, std::vector<KernelInfo*>& kernels);
  
  /**
   * findAllKernels() - Find all non-fused kernels that can compute given problem.
   * @problem: The problem to find kernels for.
   * @useP2PStore: True if kernels should use P2P RDMA stores.
   * @kernels: [OUT] A vector of kernels for each optimization level.
   *
   * Return - True if atleast one kernel is found, otherwise false. 
   */
  bool findAllKernels(KMMProblem problem, bool useP2PStore,
                      std::vector<std::vector<KernelInfo*>>& kernels);

protected:
  /**
   * findKernelForSubProblem() - Find best kernel for a sub problem 
   *                             (of single factor) from a map of opt level to kernels. 
   * @subProblem: A sub problem with n() == 1
   * @kernels: A vector of OptLevel to a vector of kernels at that level.
   * 
   * Return - The best kernel for the sub problem using FastKron's kernel search algorithm.
   */
  KernelInfo* findKernelForSubProblem(KMMProblem subProblem, const std::vector<std::vector<KernelInfo*>>& kernels);

  /**
   * findKernelAtOptLevel() - Find best kernel for a subproblem at an opt level. This function should be 
   *                          implemented by each Kernel Database.
   * @subproblem: A sub problem with n() == 1.
   * @kernelsForOptLevel: A vector of kernels at an opt level.
   *
   * Return - The best kernel found for the opt level.
   */
  virtual KernelInfo* findKernelAtOptLevel(KMMProblem subProblem, const std::vector<KernelInfo*>& kernelsForOptLevel) = 0;

  /**
   * filterFastestFusedKernels() - Filter all fused kernels to find fastest fused kernels for a problem.
   * @problem: The problem to find kernels for.
   * @kernels: The set of kernels to filter from.
   * 
   * Return - A map of number of fusion -> a vector of all fused kernels at this number of fusion.
   */
  virtual std::map<uint32_t, std::vector<KernelInfo*>, std::greater<int>> 
          filterFastestFusedKernels(const KMMProblem& problem, const std::vector<KernelInfo*>& kernels);
  /***************************************************************************/

  /**************************** Helper Functions *****************************/
protected:
  /**
   * occupancyDetails() - A pure virtual function to obtain Kernel Occupancy detail as a string. 
   *                      This function should be implemented by every KernelDatabase.
   * @kernel: The kernel to get occupancy details.
   * @problem: The problem computed by the kernel.
   *
   * Return: A string of occupancy details.
   */
  virtual std::string occupancyDetails(KernelInfo* kernelInfo, KMMProblem problem) = 0;

public:
  /**
   * getKernel() - Obtain a kernel with the given string representation.
   * @repr: string representation.
   *
   * Return: The kernel found or nullptr if no kernel found.
   */
  KernelInfo* getKernel(std::string repr);
  /***************************************************************************/
};

#include "kernel_db/kernel_db.inline.h"