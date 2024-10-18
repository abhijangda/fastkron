#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>

#if defined(ENABLE_CUDA) && defined(ENABLE_MULTI_GPU) 
  #include <nccl.h>
#endif

#include "utils/utils.h"
#include "utils/logger.h"

#include "kernel_db/cuda_kernel_db.h"
#include "kernels/cuda_kmmkernel.h"

#ifdef ENABLE_CUDA
  //Defines ALL_CUDA_KERNELS array
  #include "kernels/cuda/kron-kernels/kernel_decl.inc"
#endif

/**
 * @AllCUDAKernels: An array of All CUDA kernels compiled.
*/
CUDAKMMKernel AllCUDAKernels[] = {
#ifdef ENABLE_CUDA
  ALL_CUDA_KERNELS
#endif
};


CUDAKernelDatabase::CUDAKernelDatabase() : isDistributed_(false) {
  streams.push_back(NULL);
  //Load all CUDA kernels and check if each kernel is valid
  loadKernels<CUDAKMMKernel>(AllCUDAKernels, sizeof(AllCUDAKernels)/sizeof(CUDAKMMKernel));
  for (uint i = 0; i < sizeof(AllCUDAKernels)/sizeof(CUDAKMMKernel); i++) {
    CUDAKMMKernel& info = AllCUDAKernels[i];
    if (!info.isValid()) abort();
    CUDA_CHECK(info.setSharedMemAttr());
  }
}

CUDAKernelDatabase::~CUDAKernelDatabase() {
  streams.clear();
  if (isDistributed_) {
#ifdef ENABLE_MULTI_GPU
    for (uint g = 0; g < gpusInM_; g++) {
      int s = pthread_barrier_destroy(&barriers_[g]);
      PTHREAD_BARRIER_CHECK(s);
    }

    delete threads_;
    delete barriers_;

    if (distComm_ == DistComm::NCCL) {
      for (int i=0; i<ncclComms.size(); i++)
        ncclCommDestroy((ncclComm_t)ncclComms[i]);
    }
#endif
  }
}

fastKronError CUDAKernelDatabase::init(void* ptrToStream, int gpus,
                                       int gpusInM, int gpusInK, int gpuKrons) {
  numGPUs_ = gpus;
  setCUDAStream(ptrToStream);
  if (numGPUs_ > numDevices()) {
    return fastKronInvalidArgument;
  }

  isDistributed_ = gpus > 1;

  for (int i = 0; i < numGPUs_; i++) {
    //Get information about each GPU
    auto detail = new CUDAArchDetails(i);
    hardware.push_back(detail);
    Logger(LogLevel::Info) << "Detected GPU " << i << std::endl << (*detail);
  }

  if (isDistributed_) {
#ifdef ENABLE_MULTI_GPU
    bool allP2PAccess = true;
    for (int g1 = 0; g1 < gpus; g1++) {
      for (int g2 = 0; g2 < gpus; g2++) {
        if (g1 == g2) continue;
        int p2pAccess = -1;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&p2pAccess, g1, g2));
        if (p2pAccess == 0) {allP2PAccess = false; break;}
        CUDA_CHECK(cudaSetDevice(g1));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(g2, 0));
      }
      if (!allP2PAccess) break;
    }

    distComm_ = env::getDistComm();

    if (distComm_ == DistComm::P2P) {
      if (!allP2PAccess) {
        Logger(LogLevel::Debug) << "P2P Access among GPUs is available" << std::endl;
        distComm_ = DistComm::DistCommNone;
      }
    } else if (distComm_ == DistComm::NCCL) {
      int devs[gpus];
      distComm_ = DistComm::NCCL;
      ncclUniqueId ncclId;
      ncclGetUniqueId(&ncclId);
      Logger(LogLevel::Debug) << "Initializing NCCL"<<std::endl;
      for (int i = 0; i < gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        ncclComms.push_back(nullptr);
        devs[i] = i;
      }
      NCCLCHECK(ncclCommInitAll((ncclComm_t*)&ncclComms[0], gpus, devs));
    }

    if (distComm_ == DistComm::DistCommNone) {
      if (allP2PAccess) {
        distComm_ = DistComm::P2P;
      } else {
        int devs[gpus];
        distComm_ = DistComm::NCCL;
        ncclUniqueId ncclId;
        ncclGetUniqueId(&ncclId);
        Logger(LogLevel::Debug) << "Initializing NCCL"<<std::endl;
        for (int i = 0; i < gpus; i++) {
          CUDA_CHECK(cudaSetDevice(i));
          ncclComms.push_back(nullptr);
          devs[i] = i;
        }
        NCCLCHECK(ncclCommInitAll((ncclComm_t*)&ncclComms[0], gpus, devs));
      }
    }

    Logger(LogLevel::Info) << "Using " << distComm_ << 
                                  " for distributed communication" <<
                                  std::endl;

    if (gpusInK >= 1)
      gpusInK_ = gpusInK;
    else
      gpusInK_ = 2;//ilog2(gpus);
    
    if (gpusInM >= 1)
      gpusInM_ = gpusInM;  
    else
      gpusInM_ = 1;//ilog2(gpus);
      
    //TODO: Check that gpuKrons batch is valid, i.e., P1*P2..PBatch <= gpusInK
    if (gpuKrons > 0)
      perGPUKronBatch_ = gpuKrons;
    else 
      perGPUKronBatch_ = 1;

    //TODO: Check if gpusInK_ == 1 then perGPUKronBatch = NumKrons

    Logger(LogLevel::Debug) << "gpusInRows " << gpusInM_ <<
                 " gpusInCols " << gpusInK_ << 
                 " gpuKronBatch " << perGPUKronBatch_ <<
                 std::endl;
    if (gpusInK_ * gpusInM_ != numGPUs_)  {
      Logger(LogLevel::Info) << "gpusInCols * gpusInRows != total gpus (" << 
                   gpusInK_ * gpusInM_ << "!= " << 
                   numGPUs_<< ")" << std::endl;
      abort();
    }
    //TODO: Check that localKrons <= log (gpuK_)_P
    
    //All gpus with same row shares the same barrier
    //TODO: free
    barriers_ = new pthread_barrier_t[gpusInM_];
    threads_ = new thread_pool<ThreadArgs*>(numGPUs_);

    for (int i = 0; i < gpusInM_; i++) {
      int s = pthread_barrier_init(&barriers_[i], NULL, gpusInK_);
      PTHREAD_BARRIER_CHECK(s);
    }
#endif
  }

  return fastKronSuccess;
}

void CUDAKernelDatabase::setCUDAStream(void* ptrToStream) {
  streams.clear();
  cudaStream_t* t = new cudaStream_t;
  *t = 0;
  for (int i = 0; i < numGPUs_; i++) {
    if (ptrToStream != NULL) {
      cudaStream_t* s = new cudaStream_t;
      *s = *(((cudaStream_t*)ptrToStream) + i);
	    streams.push_back((void*)s);
    }
    else
	    streams.push_back(t);
  }
}

int CUDAKernelDatabase::numDevices() {
  int devs;
  CUDA_CHECK(cudaGetDeviceCount(&devs));
  return devs;
}

CUDAArchDetails CUDAKernelDatabase::getCUDADeviceProperties() {
  return *(dynamic_cast<CUDAArchDetails*>(hardware[0]));
}

fastKronError CUDAKernelDatabase::procMemset(uint32_t proc, Matrix& m, float val) {
  CUDA_CHECK(cudaSetDevice(proc));
  float* host = new float[m.numel()];
  memset<float>(host, m.numel(), val);
  CUDA_CHECK(cudaMemcpy(m.data(), host, m.numel()*sizeof(float),
                        cudaMemcpyHostToDevice));
  delete host;
  return fastKronSuccess;
}

fastKronError CUDAKernelDatabase::procMalloc(uint32_t proc, size_t size, void*& ptr) {
  CUDA_CHECK(cudaSetDevice(proc));
  CUDA_CHECK(cudaMalloc(&ptr, size));
  CUDA_CHECK(cudaMemset(ptr, 1, size));

  return fastKronSuccess;
}

fastKronError CUDAKernelDatabase::procFree(uint32_t proc, void* ptr) {
  CUDA_CHECK(cudaSetDevice(proc));
  CUDA_CHECK(cudaFree(ptr));
  return fastKronSuccess;
}

fastKronError CUDAKernelDatabase::initTune() {
  CUDA_CHECK(cudaSetDevice(0));
  return fastKronSuccess;
}


/**
 * invoke() - Invoke a CUDA kernel.
 * @FusedFacs: The number of fusion in the CUDA kernel
 * @kernel: kernel to invoke.
 * @problem: KMMProblem to compute.
 * @fidx: Factor index in the KMMProblem.
 * @distParams: Parameters for Distributed 
 * @eplogueParams: Parameter for Epilogue (alpha, beta, and Y)
 * @execMode: Execution mode
 */
template<typename KMMProblemT, typename EpilogueParamsT>
fastKronError invoke(CUDAKMMKernel& kernelInfo, KMMProblemT problem,
                     const uint fidx, 
                     DistributedParams distParams,
                     EpilogueParamsT epilogueParams,
                     KernelMode execMode,
                     cudaStream_t stream) {
  cudaError_t status;

  KernelParams<KMMProblemT> params (problem, nullptr, 
                                  kernelInfo.getTileX(problem), 
                                  kernelInfo.getTileF(problem), 
                                  fidx, execMode);

  FusedParams<KMMProblemT> fusedParams (problem, kernelInfo.getMaxTileX().n());
  dim3 g = kernelInfo.grid(problem);
  std::cout <<fidx << "  " << g.x << " " << g.y << " " << g.z << std::endl;
  //TODO: Change this to kernelInfo.invoke
  typedef void (*KronMatmulKernelTy)(KernelParams<KMMProblemT>&, FusedParams<KMMProblemT>&, 
                                     DistributedParams&, EpilogueParams&, 
                                     dim3, dim3, uint32_t, cudaStream_t);
  KronMatmulKernelTy(kernelInfo.kernelInvoker)(params, fusedParams, distParams, 
                                               epilogueParams, 
                                               kernelInfo.grid(problem),
                                               kernelInfo.block(),
                                               kernelInfo.getSharedMemSize(problem),
                                               stream);
  status = cudaGetLastError();
  CUDA_CHECK(status);

  return fastKronSuccess;
}

template<typename KMMProblem, typename EpilogueParams>
fastKronError CUDAKernelDatabase::invokeKernel(KMMKernel* kernel,
                                               KMMProblem problem,
                                               const uint fidx,
                                               EpilogueParams epilogueParams,
                                               KernelMode execMode) {
  DistributedParams distParams;
  cudaStream_t stream = *(cudaStream_t*)streams[0];
  CUDAKMMKernel& cudaKernel = dynamic_cast<CUDAKMMKernel&>(*kernel);

  switch(problem.n()) {
    case 1:
      return invoke(cudaKernel, problem.template factorSlice<1>(), fidx, distParams, epilogueParams,
                       execMode, stream);
    case 2:
      return invoke(cudaKernel, problem.template factorSlice<2>(), fidx, distParams, epilogueParams,
                       execMode, stream);
    case 3:
      return invoke(cudaKernel, problem.template factorSlice<3>(), fidx, distParams, epilogueParams,
                       execMode, stream);
    case 4:
      return invoke(cudaKernel, problem.template factorSlice<4>(), fidx, distParams, epilogueParams,
                       execMode, stream);
    case 5:
      return invoke(cudaKernel, problem.template factorSlice<5>(), fidx, distParams, epilogueParams,
                       execMode, stream);
    case 6:
      return invoke(cudaKernel, problem.template factorSlice<6>(), fidx, distParams, epilogueParams,
                       execMode, stream);
    default:
      Logger(LogLevel::Debug) << "Invalid number of fused kernels: " << problem.n() << std::endl;
      return fastKronKernelNotFound;
  }
}


fastKronError CUDAKernelDatabase::invokeKernel(KMMKernel* kernel, KMMProblem problem,
                                    const uint fidx,
                                    EpilogueParams epilogueParams,
                                    KernelMode execMode) {
  return invokeKernel<KMMProblem, EpilogueParams>(kernel, problem, fidx, epilogueParams, execMode);
}

fastKronError CUDAKernelDatabase::invokeKernel(KMMKernel* kernel, KMMProblemStridedBatched problem,
                                    const uint fidx,
                                    EpilogueStridedBatchedParams epilogueParams,
                                    KernelMode execMode) {
  return invokeKernel<KMMProblemStridedBatched, EpilogueStridedBatchedParams>(kernel, problem, fidx, epilogueParams, execMode);
}

fastKronError CUDAKernelDatabase::invokeP2PStoreKernel(KMMKernel* kernel, 
                                                       KMMProblem problem,
                                                       const uint fidx,  
                                                       DistributedParams distParams, 
                                                       EpilogueParams epilogueParams,
                                                       KernelMode execMode) {
  cudaStream_t stream = *(cudaStream_t*)streams[distParams.proc()];
  CUDAKMMKernel& cudaKernel = dynamic_cast<CUDAKMMKernel&>(*kernel);

  switch (problem.n()) {
    case 1:
      return invoke(cudaKernel, problem.template factorSlice<1>(), fidx, distParams, epilogueParams,
                       execMode, stream);
    case 2:
      return invoke(cudaKernel, problem.template factorSlice<2>(), fidx, distParams, epilogueParams,
                       execMode, stream);
    case 3:
      return invoke(cudaKernel, problem.template factorSlice<3>(), fidx, distParams, epilogueParams,
                       execMode, stream);
    case 4:
      return invoke(cudaKernel, problem.template factorSlice<4>(), fidx, distParams, epilogueParams,
                       execMode, stream);
    case 5:
      return invoke(cudaKernel, problem.template factorSlice<5>(), fidx, distParams, epilogueParams,
                       execMode, stream);
    case 6:
      return invoke(cudaKernel, problem.template factorSlice<6>(), fidx, distParams, epilogueParams,
                       execMode, stream);
    default:
      Logger(LogLevel::Debug) << "Invalid number of fused kernels: " << problem.n() << std::endl;
  }

  return fastKronKernelNotFound;
}

template<typename KMMProblemT, typename EpilogueParamsT>
fastKronError CUDAKernelDatabase::timeKernel(KMMKernel* kernel,
                                             KMMProblemT problem,
                                             const uint fidx, 
                                             DistributedParams distParams,
                                             EpilogueParamsT epilogueParams,
                                             KernelMode execMode, 
                                             bool useP2PStore,
                                             int warmups, int runs,
                                             float& runtime) {
#ifdef ENABLE_MULTI_GPU
//TODO: Also for FULL_TUNE
  if ((dynamic_cast<CUDAKMMKernel*>(kernel))->getLocalSize() > 0 || 
      (problem.f(0).q() >= 64 && kernel->getMaxTileF().q() <= 32)) {
    //skip probably slow kernels
    runtime = std::numeric_limits<float>::max();
    return fastKronSuccess;
  }
#endif

  cudaStream_t stream = *(cudaStream_t*)streams[0];
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t startEvent, endEvent;
  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&endEvent));
  fastKronError status = fastKronSuccess;
  for (int r = 0; r < warmups + runs; r++) {
    if (r == warmups) CUDA_CHECK(cudaEventRecord(startEvent, stream));
    if (useP2PStore) {
      status = invokeP2PStoreKernel(kernel, problem, fidx,
                                    distParams, epilogueParams, execMode);
    } else {
      status = invokeKernel(kernel, problem, fidx,
                            epilogueParams, execMode);
    }
  }
  CUDA_CHECK(cudaEventRecord(endEvent, stream));
  CUDA_CHECK(cudaEventSynchronize(endEvent));
  if (status != fastKronSuccess) {
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(endEvent));
    Logger(LogLevel::Info) << "Error in CUDA autotuning: "   <<
                              fastKronGetErrorString(status) <<
                              std::endl;
    return status;
  }
  CUDA_CHECK(cudaEventElapsedTime(&runtime, startEvent, endEvent));
  runtime = runtime/runs;
  CUDA_CHECK(cudaEventDestroy(startEvent));
  CUDA_CHECK(cudaEventDestroy(endEvent));
  return status;
}

fastKronError CUDAKernelDatabase::timeKernel(KMMKernel* kernel, KMMProblem problem, 
                                                      const uint fidx, 
                                                      DistributedParams distParams,
                                                      EpilogueParams epilogueParams,
                                                      KernelMode execMode, 
                                                      bool useP2PStore,
                                                      int warmups, int runs,
                                                      float& runtime) {
  return timeKernel<KMMProblem, EpilogueParams>(kernel, problem, fidx, distParams, epilogueParams, execMode, 
                                                useP2PStore, warmups, runs, runtime);

}

fastKronError CUDAKernelDatabase::timeKernel(KMMKernel* kernel, KMMProblemStridedBatched problem, 
                                                     const uint fidx, 
                                                     DistributedParams distParams,
                                                     EpilogueStridedBatchedParams epilogueParams,
                                                     KernelMode execMode, 
                                                     bool useP2PStore,
                                                     int warmups, int runs,
                                                     float& runtime) {
  return timeKernel<KMMProblemStridedBatched, EpilogueStridedBatchedParams>(kernel, problem, fidx, distParams, epilogueParams, 
                                                                            execMode, useP2PStore, warmups, runs, runtime);
}

std::map<uint32_t, std::vector<KMMKernel*>, std::greater<int>>
  CUDAKernelDatabase::filterFastestFusedKernels(const KMMProblem& problem, 
                                                const std::vector<KMMKernel*>& kernels) {
  //TODO: 16 for Ampere and 8 for Volta
  uint32_t MinConsecutiveStoreElems = (getCUDADeviceProperties().smArch == SMArch::ampere) ? 16 : 8;

  //A fused kernel stores logP (TK) consecutive elements.
  //Remove all kernels that stores (< MinConsecutiveStoreElems).
  std::vector<KMMKernel*> validFusedKernels;
  
  {
    auto filter = [problem, MinConsecutiveStoreElems](KMMKernel* kernel) {
      const int PpowerN = (int)powf(problem.f(0).p(), kernel->getFusedFacs());
      const int consecutiveStoreElems = kernel->getMaxTileX().n()/PpowerN;
      return consecutiveStoreElems >= MinConsecutiveStoreElems;
    };

    std::copy_if(kernels.begin(), kernels.end(), std::back_inserter(validFusedKernels), filter);
  }

  return KernelDatabase::filterFastestFusedKernels(problem, validFusedKernels);
}

/**
 * blocksPerSM() - Returns blocks per SM occupied by a CUDA kernel based on occupancy
 */
static float blocksPerSM(const CUDAArchDetails gpu, CUDAKMMKernel* kernel, dim3 grid) {
  uint32_t regOcc = gpu.regsPerSM / (kernel->block().x * kernel->getNumRegs());
  uint32_t shmemOcc = gpu.sharedMemPerSM / kernel->getMaxSharedMemSize();
  return min(min(regOcc, shmemOcc), gpu.maxBlocksPerSM);
}

template<typename KMMProblemT>
KMMKernel* CUDAKernelDatabase::findKernelAtOptLevel(KMMProblemT subProblem, 
                                                    const std::vector<KMMKernel*>& kernelsForOptLevel) {
  if (kernelsForOptLevel.size() > 0) {
    //Find kernels that have either same P or same Q
    std::vector<KMMKernel*> kernelsWithSamePOrQ;
    std::copy_if(kernelsForOptLevel.begin(), kernelsForOptLevel.end(), 
                 std::back_inserter(kernelsWithSamePOrQ),
                 [subProblem](auto& kernel){return kernel->getMaxFactor().p() == subProblem.f(0).p() or 
                                            kernel->getMaxFactor().q() == subProblem.f(0).q();});
    std::vector<KMMKernel*> filteredKernels;
    if (kernelsWithSamePOrQ.size() > 0) {
      filteredKernels = kernelsWithSamePOrQ;
    } else {
      filteredKernels = kernelsForOptLevel;
    }
    //sort kernels in descending order based on the number of thread blocks a kernel invoke
    auto order = [subProblem, this](auto k1, auto k2) {
      return ((CUDAKMMKernel*)k1)->getNumBlocks(subProblem) > 
             ((CUDAKMMKernel*)k2)->getNumBlocks(subProblem);
    };
    std::sort(filteredKernels.begin(), filteredKernels.end(), order);
    for (auto k : filteredKernels) {
      uint blocksm = blocksPerSM(getCUDADeviceProperties(), (CUDAKMMKernel*)k, ((CUDAKMMKernel*)k)->grid(subProblem));
      if (((CUDAKMMKernel*)k)->getNumBlocks(subProblem) <= getCUDADeviceProperties().numSMs * blocksm) {
        return k;
      }
    }

    //If no kernel is found then return the kernel with max reuse
    return filteredKernels[filteredKernels.size() - 1];
  }

  return nullptr;
}

template<typename KMMProblemT>
std::string CUDAKernelDatabase::occupancyDetails(KMMKernel* kernelInfo, KMMProblemT problem) {
  CUDAKMMKernel* cudaKernel = dynamic_cast<CUDAKMMKernel*>(kernelInfo);
  std::stringstream ss;
  dim3 grid = cudaKernel->grid(problem);
  dim3 block = cudaKernel->block();
  std::string indent = "  ";

  ss << indent << "Grid          : {" << grid.x << ", " << grid.y << ", " << grid.z << "}" << std::endl
     << indent << "Block         : {" << block.x << ", " << block.y << ", " << block.z << "}" << std::endl
     << indent << "Shared Mem    : " << cudaKernel->getSharedMemSize(problem) << std::endl 
     << indent << "Reg per Thread: " << cudaKernel->getNumRegs() << std::endl
     << indent << "Blocks Per SM : " << blocksPerSM(getCUDADeviceProperties(), cudaKernel, cudaKernel->grid(problem)) << std::endl
     << indent << "Local Memory  : " << cudaKernel->getLocalSize() << std::endl;

  return ss.str();
}

std::string CUDAKernelDatabase::occupancyDetails(KMMKernel* kernelInfo, KMMProblem problem) {
  return occupancyDetails<KMMProblem>(kernelInfo, problem);
}

std::string CUDAKernelDatabase::occupancyDetails(KMMKernel* kernelInfo, KMMProblemStridedBatched problem) {
  return occupancyDetails<KMMProblemStridedBatched>(kernelInfo, problem);
}

CUDAArchDetails::CUDAArchDetails(int dev) {
  cudaDeviceProp prop;

  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  numSMs             = prop.multiProcessorCount;
  maxBlocksPerSM     = prop.maxBlocksPerMultiProcessor;
  maxThreadsPerBlock = prop.maxThreadsPerBlock;
  maxThreadsPerSM    = prop.maxThreadsPerMultiProcessor;
  regsPerSM          = prop.regsPerMultiprocessor;
  maxRegsPerThread   = 256; 
  sharedMemPerSM     = prop.sharedMemPerMultiprocessor;
  sharedMemPerBlock  = prop.sharedMemPerBlock;
  name               = std::string(prop.name);
  computeMajor       = prop.major;
  computeMinor       = prop.minor;
  warpSize           = prop.warpSize;
  smArch             = computeCapabilityToSMArch(computeMajor, computeMinor);
}
