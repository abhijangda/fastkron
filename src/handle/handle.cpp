#include <cassert>
#include <iostream>

#include "utils/utils.h"
#include "utils/thread_pool.h"
#include "handle/handle.h"
#include "handle/op.h"
#include "env/env.h"
#include "autotuner/autotuner.h"
#include "kmm/kmmalgo.h"

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

FastKronHandle::FastKronHandle(uint32_t backends) :
  backends(backends), autotuner(*this)
#ifdef ENABLE_CUDA
  , cudaKernels()
#endif
{}

fastKronOp swapFastKronOp(fastKronOp op) {
  if (op == fastKronOp_N) return fastKronOp_T;
  if (op == fastKronOp_T) return fastKronOp_N;
  return fastKronOp_N;
}

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

#ifdef ENABLE_X86
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

fastKronError FastKronHandle::xgemm(KMMProblem problem, const fastKronBackend backend,
                                    void* temp1, void* temp2, EpilogueParams epilogueParams) {
  if (problem.y().data()  == nullptr || temp1 == nullptr ||
      hasBackend(backend) == false) 
      return fastKronInvalidArgument;

  if (problem.y().data() == epilogueParams.z<void>() && 
      (temp1 == nullptr || temp2 == nullptr))
      return fastKronInvalidArgument;

  void* temps[2] = {temp1, temp2};

  return xgemm(false, problem, backend, temps, epilogueParams);
}

fastKronError FastKronHandle::xgemm(bool isforward, KMMProblem problem, 
                                    const fastKronBackend backend, 
                                    void** intermediatePtrs,
                                    EpilogueParams epilogueParams) {

  if (problem.mmtype() == FastKronMMType::KMM && problem.m() == 1) {
    fastKronOp opF = swapFastKronOp(problem.opFs());
    problem = KMMProblem(FastKronMMType::MKM, problem.type(), problem.x(),
                         fastKronOp_N, problem.n(), problem.fs(), opF, problem.y());
  }

  fastKronError err;
  TunedKernelsSeries kernelSeries;

  auto kernelDb = getKernelDb(backend);

  if (canTune()) {
    //Tune for the fastest kernel series for the problem
    uint32_t Ps[problem.n()];
    uint32_t Qs[problem.n()];
    problem.ps(Ps);
    problem.qs(Qs);

    err =  autotuner.tune(problem, backend, kernelSeries);
    if (err != fastKronSuccess)
      return err;
  } 
  else {
    //Otherwise, use a low-latency algorithm to obtain an efficient kernel
    kernelSeries = kernelDb->kernelSeriesForProblem(problem);
  }

  KMMProblem::Matrices intermediates({});
  if (isforward) {
    gekmmIntermediates(problem, intermediatePtrs, intermediates);
  } else {
    void* temps[2] = {intermediatePtrs[0],intermediatePtrs[1]};

    auto kernelSeriesIter = kernelSeries.begin();

    err = getIntermediates(false, problem, temps, intermediates, kernelSeries.size(),
                           [&] (const KMMProblem)
                           {uint32_t len = kernelSeriesIter->end - kernelSeriesIter->start + 1;
                            kernelSeriesIter++; return len;}); 
  }

  auto kernelSeriesIter = kernelSeries.begin();

  //Execute GeKMM algorithm using above kernels
  err = executeGeMM(problem, intermediates,
    [&kernelSeriesIter](const KMMProblem)
      {return kernelSeriesIter->end - kernelSeriesIter->start + 1;},
    [&kernelSeriesIter, &epilogueParams, kernelDb, problem, this]
      (const KMMProblem subProblem, uint32_t rstart, KMMProblem::Matrices intermediates) {
        fastKronError err;
        auto kernel = *kernelSeriesIter;

        KMMKernel* selectedKernel = kernel.kernel;
        assert(rstart == kernel.end);
        epilogueParams.isLastFactor = kernel.end == subProblem.n()-1;
        err = kernelDb->invokeKernel(selectedKernel, subProblem,
                                     rstart, intermediates, epilogueParams,
                                     KernelModeNormal);
        kernelSeriesIter++;
        return err;
    });

  return err;
}

fastKronError FastKronHandle::xgemmStridedBatched(KMMProblemStridedBatched problem, 
                                                  const fastKronBackend backend, 
                                                  void* temp1, void* temp2,
                                                  EpilogueStridedBatchedParams epilogueParams) {
  if (problem.y().data()  == nullptr || temp1 == nullptr ||
      hasBackend(backend) == false) 
      return fastKronInvalidArgument;

  if (problem.y().data() == epilogueParams.z<void>() && 
    (temp1 == nullptr || temp2 == nullptr))
    return fastKronInvalidArgument;
  
  void* temps[] = {(void*)temp1, (void*)temp2}; 
  uint64_t strideTemps[2] = {0,0};

  {
    Matrix r, t;
    gekmmResultTemp(problem.batchProblem(0), r, t);
    strideTemps[0] = strideTemps[1] = t.numel();
  }

  return xgemmStridedBatched(false, problem, backend, temps, strideTemps, epilogueParams);
}

fastKronError FastKronHandle::xgemmStridedBatched(bool isforward, KMMProblemStridedBatched problem, 
                                                  const fastKronBackend backend, 
                                                  void** intermediatePtrs,
                                                  uint64_t strideIntermediates[],
                                                  EpilogueStridedBatchedParams epilogueParams) {
  fastKronError err = fastKronSuccess;
  TunedKernelsSeries kernelSeries;

  using StridedBatchMatrices = KMMProblemStridedBatched::Matrices;

  auto kernelDb = getKernelDb(backend);

  if (canTune()) {
    //Tune for the fastest kernel series for the problem
    err =  autotuner.tune(problem, backend, kernelSeries);
    if (err != fastKronSuccess)
      return err;
  } 
  else {
    //Otherwise, use a low-latency algorithm to obtain an efficient kernel
    kernelSeries = kernelDb->kernelSeriesForProblem(problem);
  }

  auto kernelSeriesIter = kernelSeries.begin();

  StridedBatchMatrices intermediates({});

  if (isforward) {
    gekmmIntermediates(problem, intermediatePtrs, strideIntermediates, intermediates);
  } else {
    void* temps[2] = {intermediatePtrs[0],intermediatePtrs[1]};

    auto kernelSeriesIter = kernelSeries.begin();

    err = getIntermediates(false, problem, temps, nullptr, intermediates, kernelSeries.size(),
                           [&] (const KMMProblemStridedBatched)
                           {uint32_t len = kernelSeriesIter->end - kernelSeriesIter->start + 1;
                            kernelSeriesIter++; return len;});
  }

  //Execute GeKMM algorithm using above kernels
  err = executeGeMM(problem, intermediates,
    [&kernelSeriesIter](const KMMProblemStridedBatched) 
      {return kernelSeriesIter->end - kernelSeriesIter->start + 1;},
    [&kernelSeriesIter, &epilogueParams, kernelDb, problem, this]
      (const KMMProblemStridedBatched subProblem, uint32_t rstart, StridedBatchMatrices intermediates) {
        fastKronError err;
        auto kernel = *kernelSeriesIter;

        KMMKernel* selectedKernel = kernel.kernel;
        assert(rstart == kernel.end);
        epilogueParams.isLastFactor = kernel.end == subProblem.n()-1;
        err = kernelDb->invokeKernel(selectedKernel, subProblem, 
                                     rstart, intermediates, epilogueParams,
                                     KernelModeNormal);
        kernelSeriesIter++;
        return err;
    });

  return err;
}

fastKronError FastKronHandle::gekmmIntermediates(KMMProblem problem, void* ptrs[], 
                                                 KMMProblem::Matrices& intermediates) {
#ifdef ENABLE_CUDA
  if (cudaKernels.isDistributed_) {
    if (!checkDistributedKronSizes(problem, 
                                   cudaKernels.perGPUKronBatch_,
                                   cudaKernels.gpusInK_))
      return fastKronInvalidArgument;
  }
#endif

  auto e = getIntermediates(true, problem, ptrs, intermediates, problem.n(),
                            [](const KMMProblem) {return 1;});
  return e;
}

fastKronError FastKronHandle::gekmmResultTemp(KMMProblem problem,
                                              Matrix& result, Matrix& maxTemp) {
  if (problem.n() == 1) {
    result = problem.y();
    maxTemp = (problem.x().n() > problem.y().n()) ? problem.x() : problem.y();
    return fastKronSuccess;
  }
  
  KMMProblem::Matrices intermediates ({});
  //TODO: Should move to individual backend
  fastKronError e = gekmmIntermediates(problem, nullptr, intermediates);

  result = problem.y();
  maxTemp = intermediates[0];
  // std::cout << 329 << " " << intermediates.len() << " " << std::endl;
  if (problem.n() > 1) {
    maxTemp = *std::max_element(&intermediates[0], &intermediates[problem.n()], //TODO: Write it as intermediates.begin, intermediates.end
                                [](Matrix& a, Matrix& b) {
                                  return a.n() < b.n();
                                });
  }
  
  if (e == fastKronSuccess) {
  #if defined(ENABLE_CUDA) && defined(ENABLE_MULTI_GPU)
    size_t tempCols, resultCols;
    uint gpuM = problem.m();
    getDistributedSizes(problem.m(), maxTemp.l(), gpuM, tempCols);
    getDistributedSizes(problem.m(), result.l(),  gpuM, resultCols);
    if (cudaKernels.isDistributed_ and cudaKernels.distComm_ == DistComm::NCCL)
      //Include size of send and recv buffers 
      tempCols = tempCols * 2;
    
    result  = Matrix(gpuM, resultCols);
    maxTemp = Matrix(gpuM, tempCols);
  #endif
  }

  return e;
}

fastKronError FastKronHandle::gekmmIntermediates(KMMProblemStridedBatched problem, void* ptrs[],
                                                 uint64_t strideIntermediates[],
                                                 KMMProblemStridedBatched::Matrices& intermediates) {
  auto e = getIntermediates(true, problem, ptrs, strideIntermediates, intermediates, problem.n(),
                            [](const KMMProblemStridedBatched) {return 1;});
  return e;
}

fastKronError FastKronHandle::gekmmResultTemp(KMMProblemStridedBatched problem, StridedBatchMatrix& result,
                                              StridedBatchMatrix& maxTemp) {
  if (problem.n() == 1) {
    result = problem.y();
    maxTemp = (problem.x().n() > problem.y().n()) ? problem.x() : problem.y();
    return fastKronSuccess;
  }

  KMMProblemStridedBatched::Matrices intermediates;
  //TODO: Should move to individual backend
  
  fastKronError e = gekmmIntermediates(problem, nullptr, nullptr, intermediates);

  result = problem.y();
  maxTemp = *std::max_element(&intermediates[0], &intermediates[problem.n()],
                              [](Matrix& a, Matrix& b) {
                                std::cout << 401 << " " << a.n() << " " << b.n() << std::endl;
                                return a.n() < b.n();
                              });
  
  return e;
}

fastKronError FastKronHandle::gekmmSizes(KMMProblem problem,
                                         size_t* resultSize,
                                         size_t* tempSize) {
  if (resultSize == nullptr) return fastKronInvalidArgument;
  if (tempSize   == nullptr) return fastKronInvalidArgument;

  Matrix result, maxTemp;
  
  auto e = gekmmResultTemp(problem, result, maxTemp);
  *tempSize   = maxTemp.numel();
  *resultSize = result.numel();

  return e;
}

fastKronError FastKronHandle::gekmmSizesForward(KMMProblem problem,
                                                size_t* resultSize, size_t* intermediateSizes) {
  if (resultSize == nullptr || intermediateSizes == nullptr)
    return fastKronInvalidArgument;

  KMMProblem::Matrices intermediates;
  //TODO: Should move to individual backend
  fastKronError e = gekmmIntermediates(problem, nullptr, intermediates);

  if (e != fastKronSuccess) return e;

  for (uint32_t i = 1; i < problem.n(); i++) {
    intermediateSizes[i-1] = intermediates[i].numel();
  }
  
  *resultSize = problem.y().numel();

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
