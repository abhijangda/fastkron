#include <cassert>
#include <iostream>

#include "utils/utils.h"
#include "utils/thread_pool.h"
#include "handle/handle.h"
#include "handle/op.h"
#include "env/env.h"
#include "autotuner/autotuner.h"
#include "kmm/kmmalgo.h"

FastKronHandle::FastKronHandle(uint32_t backends) :
  backends(backends), autotuner(*this)
#ifdef ENABLE_CUDA
  , cudaKernels()
#endif
{}

FastKronHandle::~FastKronHandle() {}

fastKronError FastKronHandle::initCUDABackend(void* ptrToStream, int gpus,
                                              int gpusInM, int gpusInK, 
                                              int gpuKrons) {
  if (!hasBackend(fastKronBackend_CUDA))
    return fastKronBackendNotAvailable;

#ifdef ENABLE_CUDA
  cudaKernels.init(ptrToStream, gpus, gpusInM, gpusInK, gpuKrons);

  return fastKronSuccess;
#else
  //Prevent warning
  (void)ptrToStream;
  (void)gpus;
  (void)gpusInM;
  (void)gpusInK;
  (void)gpuKrons;

  return fastKronBackendNotAvailable;
#endif
}

fastKronError FastKronHandle::initHIPBackend(void* ptrToStream) {
  if (!hasBackend(fastKronBackend_HIP))
    return fastKronBackendNotAvailable;

#ifdef ENABLE_HIP
  hipKernels.init(ptrToStream);

  return fastKronSuccess;
#else
  //Prevent warning
  (void)ptrToStream; 

  return fastKronBackendNotAvailable;
#endif
}

fastKronError FastKronHandle::initX86Backend() {
  if (!hasBackend(fastKronBackend_X86))
    return fastKronBackendNotAvailable;

#ifdef ENABLE_X8
  return fastKronSuccess;
#else
  return fastKronBackendNotAvailable;
#endif
}

fastKronError FastKronHandle::setStream(fastKronBackend backend, 
                                        void* ptrToStream) {
  if (!ptrToStream)         return fastKronInvalidArgument;
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

fastKronError FastKronHandle::xgekmm(const KMMProblem problem, 
                                     const fastKronBackend backend, 
                                     void* temp1, void* temp2,
                                     EpilogueParams epilogueParams) {
  if (problem.y().data()  == nullptr || temp1 == nullptr ||
      hasBackend(backend) == false) 
      return fastKronInvalidArgument;

  if (problem.y().data() == epilogueParams.z<void>() && 
      (temp1 == nullptr || temp2 == nullptr))
      return fastKronInvalidArgument;

  fastKronError err;
  TunedKernelsSeries kernelSeries;

  void* temps[2] = {temp1, temp2};
  auto kernelDb = getKernelDb(backend);

  if (canTune()) {
    //Tune for the fastest kernel series for the problem
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
    //Otherwise, use a low-latency algorithm to obtain an efficient kernel
    kernelSeries = kernelDb->kernelSeriesForProblem(problem);
  }

  auto kernelSeriesIter = kernelSeries.begin();

  //Execute GeKMM algorithm using above kernels
  err = executeGeKMM(problem, temps, kernelSeries.size(),
    [&kernelSeriesIter](const KMMProblem) 
      {return kernelSeriesIter->kernel->getFusedFacs();},
    [&kernelSeriesIter, epilogueParams, kernelDb, this]
      (const KMMProblem subProblem, uint32_t rstart, void*[2], Matrix) {
        fastKronError err;
        auto kernel = *kernelSeriesIter;

        KMMKernel* selectedKernel = kernel.kernel;
        assert(rstart == kernel.end);
        err = kernelDb->invokeKernel(selectedKernel, subProblem, 
                                     rstart, epilogueParams,
                                     KernelModeNormal);
        kernelSeriesIter++;
        return err;
    });

  return err;
}

fastKronError FastKronHandle::gekmmResultTemp(KMMProblem problem, 
                                              Matrix& result,
                                              Matrix& temp) {
#ifdef ENABLE_CUDA
  if (cudaKernels.isDistributed_) {
    if (!checkDistributedKronSizes(problem, 
                                   cudaKernels.perGPUKronBatch_,
                                   cudaKernels.gpusInK_))
      return fastKronInvalidArgument;
  }
#endif

  uint32_t tempCols = 0;
  uint32_t resultCols = 0;

  auto e = executeGeKMM(problem, nullptr, 0,
    [](const KMMProblem) {return 1;},
    [&tempCols, &resultCols]
    (const KMMProblem kmm, uint32_t, void*[2], Matrix) {
      tempCols = std::max(tempCols, std::max(kmm.k(), kmm.l()));
      resultCols = kmm.l();
      return fastKronSuccess;
    });
  
  uint gpuM;

#if defined(ENABLE_CUDA) && defined(ENABLE_MULTI_GPU)
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

fastKronError FastKronHandle::gekmmSizes(KMMProblem problem,
                                         size_t* resultSize,
                                         size_t* tempSize) {
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

#ifdef ENABLE_MULTI_GPU
void FastKronHandle::getDistributedSizes(uint M, uint K, 
                                         uint& gpuM, uint& gpuK) {
  //TODO: Should move to individual backends
  gpuM = M/cudaKernels.gpusInM_;
  gpuK = K/cudaKernels.gpusInK_;
}
#endif
