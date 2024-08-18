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

struct FastKronHandle {
  void* result_;
  uint32_t backends;
  Autotuner autotuner;
#ifdef ENABLE_CUDA
  CUDAKernelDatabase cudaKernels;
#endif
#ifdef ENABLE_HIP
  HIPKernelDatabase hipKernels;
#endif
#ifdef ENABLE_X86
  X86KernelDatabase x86Kernels;
#endif
  uint32_t options;

  FastKronHandle(uint32_t backends);
  ~FastKronHandle();

  bool hasBackend(fastKronBackend backend) {
    return (backends & backend);
  }

  KernelDatabase* getKernelDb(fastKronBackend backend) {
    switch (backend) {
      case fastKronBackend_X86:
        #ifdef ENABLE_X86
          return &x86Kernels;
        #endif
      case fastKronBackend_CUDA:
        #ifdef ENABLE_CUDA
          return &cudaKernels;
        #endif
      case fastKronBackend_HIP:
        #ifdef ENABLE_HIP
          return &hipKernels;
        #endif
      default:
        return nullptr;
    }
  }

  std::vector<KernelDatabase*> getAllKernelDbs() {
    std::vector<KernelDatabase*> out;
    if        (hasBackend(fastKronBackend_X86))  {
      out.push_back(getKernelDb(fastKronBackend_X86));
    } else if (hasBackend(fastKronBackend_CUDA)) {
      out.push_back(getKernelDb(fastKronBackend_CUDA));
    } else if (hasBackend(fastKronBackend_HIP))  {
      out.push_back(getKernelDb(fastKronBackend_HIP));
    }
    return out;
  }

  fastKronError initX86Backend();
  fastKronError initCUDABackend(void* ptrToStream, int gpus, 
                                int gpusInM, int gpusInK, int gpuKrons);
  fastKronError initHIPBackend(void* ptrToStream);
  fastKronError setStream(fastKronBackend backends, void* ptrToStream);

  void setOptions(uint32_t options) {this->options = options;}
  bool canTune() {
    return (options & fastKronOptionsTune) ==
            fastKronOptionsTune;
  }

  bool getUseFusion() {
    return (options & fastKronOptionsUseFusion) ==
            fastKronOptionsUseFusion;
  }

  fastKronError xgekmm(const KMMProblem problem, const fastKronBackend backend,
                       void* temp1, void* temp2, EpilogueParams epilogueParams);
  fastKronError gekmmSizes(KMMProblem problem, size_t* resultSize, size_t* tempSize);
  fastKronError gekmmResultTemp(KMMProblem problem, Matrix& result, Matrix& temp);

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