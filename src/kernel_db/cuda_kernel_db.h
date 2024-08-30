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
  virtual fastKronError invokeKernel(KernelInfo* kernel, KMMProblem problem,
                                     const uint fidx,
                                     EpilogueParams epilogueParams,
                                     KernelMode execMode);
  /**
   * invokeP2PStoreKernel() - Overriding KernelDatabase::invokeP2PStoreKernel
   */
  virtual fastKronError invokeP2PStoreKernel(KernelInfo* kernel, KMMProblem problem,
                                             const uint fidx,  
                                             DistributedParams distParams, 
                                             EpilogueParams epilogueParams,
                                             KernelMode execMode);
  /**
   * timeKernel() - Overriding KernelDatabase::timeKernel
   */
  virtual fastKronError timeKernel(KernelInfo* kernel, KMMProblem problem, 
                                   const uint fidx, 
                                   DistributedParams distParams,
                                   EpilogueParams epilogueParams,
                                   KernelMode execMode, 
                                   bool useP2PStore,
                                   int warmups, int runs,
                                   float& runtime);

  /*********************** Kernel Search Methods ***************************/
protected:
  /**
   * filterFastestFusedKernels() - Overriding KernelDatabase::filterFastestFusedKernels
   */ 
  virtual std::map<uint32_t, std::vector<KernelInfo*>, std::greater<int>> 
          filterFastestFusedKernels(const KMMProblem& problem, const std::vector<KernelInfo*>& kernels);

  /**
   * findKernelAtOptLevel() - Overriding KernelDatabase::findKernelAtOptLevel
   */
  virtual KernelInfo* findKernelAtOptLevel(KMMProblem subProblem,
                                           const std::vector<KernelInfo*>& kernels);
  /***************************************************************************/

  /**************************** Helper Methods *****************************/
protected:
  /**
   * occupancyDetails() - Overriding KernelDatabase::occupancyDetails.
   */ 
  virtual std::string occupancyDetails(KernelInfo* kernelInfo, KMMProblem problem);
  /***************************************************************************/
};