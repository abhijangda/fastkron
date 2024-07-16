#include <cassert>
#include <iostream>

#include "utils/utils.h"
#include "utils/thread_pool.h"
#include "handle/handle.h"
#include "handle/op.h"
#include "env/env.h"
#include "autotuner/autotuner.h"
#include "kmm/kmmalgo.h"

#define NULL_CHECK(x) if ((x) == nullptr) return fastKronInvalidArgument;

std::string fastKronOpToStr(const fastKronOp& op) {
  switch (op) {
    case fastKronOp_N:
      return "N";
    case fastKronOp_T:
      return "T";
  }

  return NULL;
}

std::ostream& operator<<(std::ostream& os, const fastKronOp& op) {
  os << fastKronOpToStr(op);
  return os;
}

fastKronError FastKronHandle::xgekmm(const KMMProblem problem, const fastKronBackend backend, 
                                     void* temp1, void* temp2,
                                     EpilogueParams epilogueParams) {
  if (problem.y().data() == nullptr) return fastKronInvalidArgument;
  if (temp1              == nullptr) return fastKronInvalidArgument;
  if (not hasBackend(backend))       return fastKronInvalidArgument;
  fastKronError err;

  void* temps[2] = {temp1, temp2};

  auto kernelDb = getKernelDb(backend);

  TunedKernelsSeries kernelSeries;

  if (canTune()) {
    uint32_t Ps[problem.n()];
    uint32_t Qs[problem.n()];
    problem.ps(Ps);
    problem.qs(Qs);

    err =  autotuner.tune(KMMProblem(problem.type(), problem.m(), problem.n(),
                                      Ps, Qs, problem.opX(), problem.opFs()),
                              backend, kernelSeries);
    if (err != fastKronSuccess)
      return err;
  } 
  else {
    kernelSeries = kernelDb->kernelSeriesForProblem(problem);
  }

  auto kernelSeriesIter = kernelSeries.begin();

  err = executeGeKMM(problem, temps, kernelSeries.size(),
    [&kernelSeriesIter](const KMMProblem) {return kernelSeriesIter->kernel->FusedFacs;},
    [&kernelSeriesIter, epilogueParams, kernelDb, this]
      (const KMMProblem subProblem, int rstart, void* temps[2], Matrix result) {
        fastKronError err;
        auto kernel = *kernelSeriesIter;

        KernelInfo* selectedKernel = kernel.kernel;
        assert(rstart == kernel.end);
        err = kernelDb->invokeKernel(selectedKernel, rstart, 
                                     subProblem, epilogueParams,
                                     KernelModeNormal);
        kernelSeriesIter++;
        return err;
    });

  return err;
}

fastKronError FastKronHandle::gekmmResultTemp(KMMProblem problem, Matrix& result, Matrix& temp) {
#ifdef ENABLE_CUDA
  if (cudaKernels.isDistributed_) {
    if (!checkDistributedKronSizes(problem, cudaKernels.perGPUKronBatch_, cudaKernels.gpusInK_))
      return fastKronInvalidArgument;
  }
#endif

  uint32_t tempCols = 0;
  uint32_t resultCols = 0;
  auto e = executeGeKMM(problem, nullptr, 0,
    [](const KMMProblem kmm) {return 1;},
    [&tempCols, &resultCols]
    (const KMMProblem kmm, int rstart, void* temps[2], Matrix result) {
      tempCols = std::max(tempCols, std::max(kmm.k(), kmm.l()));
      resultCols = kmm.l();
      return fastKronSuccess;
    });
  
  uint gpuM;

#ifdef ENABLE_CUDA
  if (cudaKernels.isDistributed_) {
    getDistributedSizes(problem.m(), tempCols,   gpuM, tempCols);
    getDistributedSizes(problem.m(), resultCols, gpuM, resultCols);
  } else
#endif 
  {
    gpuM = problem.m();
  }

  result = Matrix(gpuM, resultCols);
  temp = Matrix(gpuM, tempCols);
  return e;
}

fastKronError FastKronHandle::gekmmSizes(KMMProblem problem, size_t* resultSize, size_t* tempSize) {
  if (resultSize == nullptr) return fastKronInvalidArgument;
  if (tempSize   == nullptr) return fastKronInvalidArgument;

  Matrix result, temp;
  //TODO: Should move to individual backend
  fastKronError e = gekmmResultTemp(problem, result, temp);

  if (e == fastKronSuccess) {
    *tempSize   = temp.numel();
#ifdef ENABLE_CUDA
    if (cudaKernels.isDistributed_ and cudaKernels.distComm_ == DistComm::NCCL)
      //Include size of send and recv buffers 
      *tempSize = (*tempSize) * 2;
#endif
    *resultSize = result.numel();
  }

  return e;
}

fastKronError FastKronHandle::initCUDABackend(void* ptrToStream, int gpus, int gpusInM, int gpusInK, int gpuKrons) {
  if (!hasBackend(fastKronBackend_CUDA)) return fastKronInvalidArgument;
#ifdef ENABLE_CUDA
  cudaKernels.init(ptrToStream, gpus, gpusInM, gpusInK, gpuKrons);
  return fastKronSuccess;
#else
  return fastKronBackendNotAvailable;
#endif
}

fastKronError FastKronHandle::initHIPBackend(void* ptrToStream) {
  if (!hasBackend(fastKronBackend_HIP)) return fastKronInvalidArgument;
#ifdef ENABLE_HIP
  hipKernels.init(ptrToStream);
  return fastKronSuccess;
#else
  return fastKronBackendNotAvailable;
#endif
}

fastKronError FastKronHandle::initX86Backend() {
  if (!hasBackend(fastKronBackend_X86)) return fastKronInvalidArgument;
#ifdef ENABLE_X86
  x86Kernels.init();
  return fastKronSuccess;
#else
  return fastKronBackendNotAvailable;
#endif
}

fastKronError FastKronHandle::setStream(fastKronBackend backend, void* ptrToStream) {
  if (ptrToStream == NULL) return fastKronInvalidArgument;
  if (!hasBackend(backend)) return fastKronBackendNotAvailable;
  
  if (backend == fastKronBackend_CUDA) {
#if ENABLE_CUDA
  cudaKernels.setCUDAStream(ptrToStream);
#else
  return fastKronBackendNotAvailable;
#endif
  } else if (backend == fastKronBackend_HIP) {
#if ENABLE_HIP
  hipKernels.setHIPStream(ptrToStream);
#else
  return fastKronBackendNotAvailable;
#endif
  }

  return fastKronSuccess;
}

// fastKronError FastKronHandle::initBackends() {
//   fastKronError err = fastKronSuccess;
  
//   if (hasBackend(fastKronBackend_X86)) 
//     err = initX86Backend();
  
//   if (err != fastKronSuccess &&
//       hasBackend(fastKronBackend_CUDA)) 
//     err = initCUDABackend();
  
//   if (err != fastKronSuccess &&
//       hasBackend(fastKronBackend_HIP)) 
//     err = initHIPBackend();

//   return err;
// }

FastKronHandle::FastKronHandle(uint32_t backends) : backends(backends),
autotuner(*this)
#ifdef ENABLE_CUDA
  , cudaKernels()
#endif
{
  //TODO: Support both modes. Single Process multi gpu and multi process multi gpu
}

void FastKronHandle::free() {
#ifdef ENABLE_CUDA
  cudaKernels.free();
#endif
}

void FastKronHandle::getDistributedSizes(uint M, uint K, uint& gpuM, uint& gpuK) {
  //TODO: Should move to individual backends
#ifdef ENABLE_CUDA  
  gpuM = M/cudaKernels.gpusInM_;
  gpuK = K/cudaKernels.gpusInK_;
#endif
}
