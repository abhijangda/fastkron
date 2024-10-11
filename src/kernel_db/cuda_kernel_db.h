#include <functional>
#include <vector>
#include <map>

#include "kmm/kmmalgo.h"
#include "kernels/params.h"
#include "kernel_db/kernel_db.h"
#include "env/env.h"
#include "utils/thread_pool.h"
#include "kernels/hw_details.h"

#pragma once

struct ThreadArgs;

/**
 * CUDAKernelDatabase executes CUDA kernels and is a subclass of KernelDatabase.
 */
class CUDAKernelDatabase : public KernelDatabase {
  //TODO: These fields should be private
public:

  /**
   * @numGPUs: Number of NVIDIA GPUs available to this CUDA database.
   */
  uint32_t numGPUs_;

  /**
   * @streams: is a vector of stream of the same size as @numGPUs; 
   */
  std::vector<void*> streams;

  /**
   * TODO: Add comments for following Multi GPU members 
   */
  uint gpusInM_;
  uint gpusInK_;
  uint perGPUKronBatch_;
  bool isDistributed_;
  DistComm distComm_;
  std::vector<void*> ncclComms;
  pthread_barrier_t* barriers_;
  thread_pool<ThreadArgs*>* threads_;

public:
  CUDAKernelDatabase();
  ~CUDAKernelDatabase();

  /**
   * init() - Initialize CUDAKernelDatabase using streams and number of gpus.
   * @ptrToStream: Pointer to an array of stream of same size as @gpus.
   * @gpus: Number of GPUs per process that CUDAKernelDatabase can use.
   * @gpusInM, @gpusInK: Number of GPUs among Row and Cols of X, Y, and Z.
   * @gpuKrons: Number of local GPU iterations to perform before communicating results among all GPUs.
   *
   * Return - fastKronSuccess if successful otherwise an error
   */
  fastKronError init(void* ptrToStream, int gpus, int gpusInM, int gpusInK, int gpuKrons);

  /**
   * setCUDAStream() - Set CUDA stream for all GPUs.
   * @ptrToStream: Pointer to an array of stream of same size as number of GPUs of this database.
   */
  void setCUDAStream(void* ptrToStream);

private:

  /**
   * numDevices() - Returns number of devices accessible to the process.
   */
  static int numDevices();

  /**
   * getCUDADeviceProperties() - Returns GPU device properties of the first GPU.
   */
  CUDAArchDetails getCUDADeviceProperties();

  /*********************** Memory Allocation Methods ***********************/
public:
  /**
   * procMemset() - Overriding KernelDatabase::procMemset.
   */
  virtual fastKronError procMemset(uint32_t proc, Matrix& m, float val);

protected:
  /**
   * procMalloc() - Overriding KernelDatabase::procMalloc.
   */
  virtual fastKronError procMalloc(uint32_t proc, size_t size, void*& ptr);

  /**
   * procFree() - Overriding KernelDatabase::procFree.
   */
  virtual fastKronError procFree(uint32_t proc, void* ptr);
  /***************************************************************************/

  /************************* Auto tuning Methods ***************************/

public:
  /**
   * initTune() - Overriding KernelDatabase::initTune.
   */
  virtual fastKronError initTune();
  /***************************************************************************/

  /**
   * invokeKernel() - Overriding KernelDatabase::invokeKernel
   */
private:
  template<typename KMMProblem, typename EpilogueParams>
  fastKronError invokeKernel(KMMKernel* kernel, KMMProblem problem,
                             const uint fidx,
                             EpilogueParams epilogueParams,
                             KernelMode execMode);
public:
  virtual fastKronError invokeKernel(KMMKernel* kernel, KMMProblem problem,
                                     const uint fidx,
                                     EpilogueParams epilogueParams,
                                     KernelMode execMode);
  virtual fastKronError invokeKernel(KMMKernel* kernel, KMMProblemStridedBatched problem,
                                     const uint fidx,
                                     EpilogueStridedBatchedParams epilogueParams,
                                     KernelMode execMode);
  /**
   * invokeP2PStoreKernel() - Overriding KernelDatabase::invokeP2PStoreKernel
   */
  virtual fastKronError invokeP2PStoreKernel(KMMKernel* kernel, KMMProblem problem,
                                             const uint fidx,  
                                             DistributedParams distParams, 
                                             EpilogueParams epilogueParams,
                                             KernelMode execMode);
  virtual fastKronError invokeP2PStoreKernel(KMMKernel* kernel, KMMProblemStridedBatched problem,
                                             const uint fidx,  
                                             DistributedParams distParams, 
                                             EpilogueStridedBatchedParams epilogueParams,
                                             KernelMode execMode)
                                             {/*P2P not supported for stridedbatched*/}
  /**
   * timeKernel() - Overriding KernelDatabase::timeKernel
   */
  virtual fastKronError timeKernel(KMMKernel* kernel, KMMProblem problem, 
                                   const uint fidx, 
                                   DistributedParams distParams,
                                   EpilogueParams epilogueParams,
                                   KernelMode execMode, 
                                   bool useP2PStore,
                                   int warmups, int runs,
                                   float& runtime);
  virtual fastKronError timeKernel(KMMKernel* kernel, KMMProblemStridedBatched problem, 
                                   const uint fidx, 
                                   DistributedParams distParams,
                                   EpilogueStridedBatchedParams epilogueParams,
                                   KernelMode execMode, 
                                   bool useP2PStore,
                                   int warmups, int runs,
                                   float& runtime);
private:
  template<typename KMMProblemT, typename EpilogueParamsT>
  fastKronError timeKernel(KMMKernel* kernel, KMMProblemT problem, 
                           const uint fidx, 
                           DistributedParams distParams,
                           EpilogueParamsT epilogueParams,
                           KernelMode execMode, 
                           bool useP2PStore,
                           int warmups, int runs,
                           float& runtime);

  /*********************** Kernel Search Methods ***************************/
protected:
  /**
   * filterFastestFusedKernels() - Overriding KernelDatabase::filterFastestFusedKernels
   */ 
  virtual std::map<uint32_t, std::vector<KMMKernel*>, std::greater<int>> 
          filterFastestFusedKernels(const KMMProblem& problem, const std::vector<KMMKernel*>& kernels);

  /**
   * findKernelAtOptLevel() - Overriding KernelDatabase::findKernelAtOptLevel
   */
  virtual KMMKernel* findKernelAtOptLevel(KMMProblem subProblem,
                                          const std::vector<KMMKernel*>& kernels) {
    return findKernelAtOptLevel<KMMProblem>(subProblem, kernels);
  }
  virtual KMMKernel* findKernelAtOptLevel(KMMProblemStridedBatched subProblem, 
                                          const std::vector<KMMKernel*>& kernels) {
    return findKernelAtOptLevel<KMMProblemStridedBatched>(subProblem, kernels);
  }

private:
  /**
   * template<typename KMMProblemT> findKernelAtOptLevel - Implementation of findKernelAtOptLevel
   */
  template<typename KMMProblemT>
  KMMKernel* findKernelAtOptLevel(KMMProblemT subProblem,
                                  const std::vector<KMMKernel*>& kernels);
  /***************************************************************************/

  /**************************** Helper Methods *****************************/
protected:
  /**
   * occupancyDetails() - Overriding KernelDatabase::occupancyDetails.
   */
  template<typename KMMProblemT>
  std::string occupancyDetails(KMMKernel* kernelInfo, KMMProblemT problem);

  virtual std::string occupancyDetails(KMMKernel* kernelInfo, KMMProblem problem);
  virtual std::string occupancyDetails(KMMKernel* kernelInfo, KMMProblemStridedBatched problem);
  /***************************************************************************/
};