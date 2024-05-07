#include <vector>
#include <unordered_map>

#include "fastkron.h"
#include "utils/thread_pool.h"
#include "env/env.h"
#include "kmm/kmmalgo.h"

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

enum ProcType {
  ProcNone = 0,
  SingleThread,
  MultipleThread,
  MultipleProcess
};

struct FastKronHandle {
  void* result_;

  FastKronHandle(uint32_t backends);

  uint32_t backends;
  bool hasBackend(fastKronBackend backend) {
    return (backends & backend);
  }

  // fastKronError initBackends();

#ifdef ENABLE_CUDA
  CUDAKernelDatabase cudaKernels;
#endif
#ifdef ENABLE_HIP
  HIPKernelDatabase hipKernels;
#endif
#ifdef ENABLE_X86
  X86KernelDatabase x86Kernels;
#endif

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

  fastKronError initX86Backend();
  fastKronError initCUDABackend(void* ptrToStream, int gpus, int gpusInM, int gpusInK, int gpuKrons);
  fastKronError initHIPBackend(void* ptrToStream);

  //TODO: these two functions should be a part of utils?
  fastKronError allocDistributedX(void* dX[], void* hX, uint M, uint K);
  fastKronError gatherDistributedY(void* dY[], void* hY, uint M, uint K, 
                                 uint NumKronMats, uint KronMatCols[], uint KronMatRows[]);

  void getDistributedSizes(uint M, uint K, uint& gpuM, uint& gpuK);
  void free();

  //Options
  bool useFusion_;
  void setUseFusion(bool v) {useFusion_ = v;}
  bool getUseFusion()       {return useFusion_;}

  TunedKernelsSeries tunedKernelSeries;
  
  //SlicedMulShape maxCompiledColsA(SlicedMulShape shape);
  //KernelInfo selectKernel(SlicedMulShape shape);
  //uint maxFusedKernels(SlicedMulShape shape);

  fastKronError xgekmm(const KMMProblem problem, const fastKronBackend backend, void* temp1, void* temp2, 
                       EpilogueParams epilogueParams);

  fastKronError distributedsgekmm(const uint NumKronMats, float* x[], float* kronMats[], float* result[],
                                  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
                                  void* streams);
  fastKronError gekmmSizes(KMMProblem problem, size_t* resultSize, size_t* tempSize);
  fastKronError gekmmResultTemp(KMMProblem problem, Matrix& result, Matrix& temp);

  //TunedKernelsSeries selectKernelSeries(const uint NumKronMats,
  //                                    uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
  //                                    bool distributedKernel);
};