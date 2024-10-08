#include <vector>
#include <unordered_map>

#include "fastkron.h"
#include "utils/thread_pool.h"
#include "env/env.h"
#include "kmm/kmmalgo.h"
#include "autotuner/autotuner.h"

#ifdef ENABLE_X86
#include "kernel_db/cpu_kernel_db.h"
#endif

#ifdef ENABLE_CUDA
#include "kernel_db/cuda_kernel_db.h"
#endif

#ifdef ENABLE_HIP
#include "kernel_db/hip_kernel_db.h"
#endif

#pragma once

class FastKronHandle {
  private:

  #ifdef ENABLE_MULTI_GPU
  public:
  #endif
  /**
   * FastKronHandle::backends - Store a set of all initialized backends as a
   * bitwise OR of enum fastKronBackends 
   */
  uint32_t backends;

  /**
   * FastKronHandle::options - Store set of options as bitwise OR of 
   * enum fastKronOptions
   */
  uint32_t options;

  /**
   * FastKronHandle::autotuner - The autotuner object to tune kernels
   */
  Autotuner autotuner;

  /**
   * FastKronHandle::cudaKernels , hipKernels, x86Kernels - Kernel Database 
   * for all backendss
   */
#ifdef ENABLE_CUDA
  CUDAKernelDatabase cudaKernels;
#endif
#ifdef ENABLE_HIP
  HIPKernelDatabase hipKernels;
#endif
#ifdef ENABLE_X86
  X86KernelDatabase x86Kernels;
#endif

  public:
  /**
   * FastKronHandle::FastKronHandle() - Initialize FastKronHandle for given backends
   * @backends: A set of fastKronBackends as a bitwise OR (||)
   *
   */
  FastKronHandle(uint32_t backends);

  /**
   * FastKronHandle::~FastKronHandle - Destroy FastKronHandle and frees all memory
   */
  ~FastKronHandle();

  /**
   * FastKronHandle::hasBackend() - Check if FastKron supports a backend
   * @backend: The backend to check
   *
   * Return - True if FastKron supports the backend otherwise no
   */
  bool hasBackend  (fastKronBackend backend);

  /**
   * FastKronHandle::setOptions() - Set options in a bitwise OR fastKronOptions
   * @options: A set of fastKronOptions as bitwise OR (||)
   *
   * Return - void
   */
  void setOptions  (uint32_t        options);

  /**
   * FastKronHandle::canTune() - Determine if tuned option is set
   * 
   * Return - True if tuned option is set otherwise false
   */
  bool canTune     ();

  /**
   * FastKronHandle::getUseFusion() - Determine if fused option is set
   * 
   * Return - True if fusion is set otherwise false
   */
  bool getUseFusion();

  /**
   * FastKronHandle::getKernelDb() - Get pointer to kernel database for backend
   * @backend: fastkronBackend to get database for
   *
   * Return - The kernel database for backend if the FastKronHandle is 
   *          initialized with the handle otherwise nullptr
   */
  KernelDatabase* getKernelDb(fastKronBackend backend);

  /**
   * FastKronHandle::getAllKernelDbs() - Get a vector of all kernel databases of initialized backends
   *
   * Return - A vector of kernel database
   */
  std::vector<KernelDatabase*> getAllKernelDbs();

  /**
   * FastKronHandle::initX86Backend() - Initialize x86 backend
   *
   * Return - fastKronSuccess if initialization is successfull
   *          fastKronBackendNotAvailable if the handle is not initialized or not compiled with x86 backend is
   */
  fastKronError initX86Backend();

  /**
   * FastKronHandle::initCUDABackend() - Initialize CUDA backend
   * @ptrToStream: A pointer to the CUDA stream
   *
   * Return - fastKronSuccess if initialization is successfull
   *          fastKronBackendNotAvailable if the handle is not initialized or not compiled with x86 backend is
   */
  //TODO: also make CUDAMgBackend
  fastKronError initCUDABackend(void* ptrToStream, int gpus, 
                                int gpusInM, int gpusInK, int gpuKrons);

  /**
   * FastKronHandle::initHIPBackend() - Initialize HIP backend
   * @ptrToStream: A pointer to the HIP stream
   *
   * Return - fastKronSuccess if initialization is successfull
   *          fastKronBackendNotAvailable if the handle is not initialized or not compiled with x86 backend is
   */
  fastKronError initHIPBackend(void* ptrToStream);

  /**
   * FastKronHandle::setStream() - Set CUDA/HIP stream
   * @backends: One of fastKronBackend_CUDA or fastKronBackend_HIP
   * @ptrToStream: A pointer to the CUDA/HIP stream
   *
   * Return - fastKronSuccess if initialization is successfull
   *          fastKronBackendNotAvailable if the handle is not initialized or not compiled with x86 backend is
   */
  fastKronError setStream(fastKronBackend backends, void* ptrToStream);

  /**
   * FastKronHandle::xgekmm() - Perform GeKMM
   * @problem: The GeKMM problem as an object of KMMProblem
   * @backend: the `fastKronBackend` of kernels
   * @temp1: Temporary array to use
   * @temp2: Temporary array to use
   * @epilogueParams: Epilogue parameters
   *
   * Return - fastKronError representing the error occurred in the operation
   */
  fastKronError xgekmm(const KMMProblem problem, const fastKronBackend backend,
                       void* temp1, void* temp2, EpilogueParams epilogueParams);


  fastKronError xgekmmStridedBatched(const KMMProblemStridedBatched problem, 
                                     const fastKronBackend backend,
                                     void* temp1, void* temp2,
                                     EpilogueStridedBatchedParams epilogueParams);
  /**
   * FastKronHandle::gekmmSizes() - Obtain GeKMM result and temporary sizes
   * @problem: The GeKMM problem as an object of KMMProblem
   * @resultSize: [OUT] number of elements of result matrix
   * @tempSize: [OUT] number of elements of a temporary matrix
   *
   * Return - fastKronError representing the error in the operation
   */
  fastKronError gekmmSizes(KMMProblem problem, size_t* resultSize, size_t* tempSize);

  /**
   * FastKronHandle::gekmmResultTemp() - Initialize GeKMM result and temporary matrix with shapes
   * @problem: The GeKMM problem as an object of KMMProblem
   * @result: [OUT] The result matrix 
   * @temp: [OUT] The temporary matrix
   *
   * Return - fastKronError representing the error in the operation
   */
  fastKronError gekmmResultTemp(KMMProblem problem, Matrix& result, Matrix& temp);
  fastKronError gekmmResultTemp(KMMProblemStridedBatched problem, 
                                StridedBatchMatrix& result, StridedBatchMatrix& temp);

  #ifdef ENABLE_MULTI_GPU
  void getDistributedSizes(uint M, uint K, uint& gpuM, uint& gpuK);
  #endif
  //TODO: these two functions should be a part of utils?
  fastKronError allocDistributedX(void* dX[], void* hX, uint M, uint K);
  fastKronError gatherDistributedY(void* dY[], void* hY, uint M, uint K, 
                                 uint NumKronMats, uint KronMatCols[], uint KronMatRows[]);
  fastKronError distributedsgekmm(const uint NumKronMats, float* x[], float* kronMats[], float* result[],
                                  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
                                  void* streams);
};

#include "handle/handle.inline.h"