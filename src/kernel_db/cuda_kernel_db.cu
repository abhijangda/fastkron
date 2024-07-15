#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>

#include <nccl.h>

#include "utils/utils.h"
#include "utils/logger.h"

#include "kernel_db/cuda_kernel_db.h"
#include "kernels/cuda_kernel_info.h"

#ifdef ENABLE_CUDA
  #include "kernels/cuda/kron-kernels/kernel_decl.inc"
#endif

CUDAKernel AllCUDAKernels[] = {
#ifdef ENABLE_CUDA
  ALL_CUDA_KERNELS
#endif
};

std::size_t std::hash<std::pair<Factor, uint>>::operator()(const std::pair<Factor, uint>& m) const {
  return hash<uint>()(m.second) ^ hash<Factor>()(m.first);
}

CUDAKernelDatabase::CUDAKernelDatabase() : isDistributed_(false) {
  streams.push_back(NULL);
  loadKernels<CUDAKernel>(AllCUDAKernels, sizeof(AllCUDAKernels)/sizeof(CUDAKernel));
  for (uint i = 0; i < sizeof(AllCUDAKernels)/sizeof(CUDAKernel); i++) {
    CUDAKernel& info = AllCUDAKernels[i];
    if (!info.isValid()) abort();
    CUDA_CHECK(info.setSharedMemAttr());
  }
  //TODO: initialize in constructor
  // fastestKernelForShape.init(this, shapeToKernelStr);
  //TODO: Check that if distP2PStore is needed then there is a kernel that can 
  //do it
  //TODO: Add if debug
}

void CUDAKernelDatabase::free() {
  streams.clear();
  if (isDistributed_) {
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
  }
}

fastKronError CUDAKernelDatabase::initTune() {
  CUDA_CHECK(cudaSetDevice(0));
  return fastKronSuccess;
}

//Launch cuda kernels
template<uint FusedFacs>
fastKronError invoke(CUDAKernel& kernelInfo, const uint kronIndex, 
                     KMMProblem problem,
                     DistributedParams distParams,
                     EpilogueParams epilogueParams,
                     KernelMode execMode,
                     cudaStream_t stream) {
  cudaError_t status;
  //Create the grid and thread block
  KernelParams<FusedFacs> params (problem, nullptr, kernelInfo.getTileX(problem), 
                                  kernelInfo.getTileF(problem), 
                                  kronIndex, execMode);
  FusedParams<FusedFacs> fusedParams (problem, kernelInfo.tileX.n());

  //Call kernel
  typedef void (*KronMatmulKernelTy)(KernelParams<FusedFacs>, FusedParams<FusedFacs>, 
                                     DistributedParams, EpilogueParams, dim3, dim3, uint32_t, cudaStream_t);
  KronMatmulKernelTy(kernelInfo.invokerFunc)(params, fusedParams, distParams, 
                                        epilogueParams, kernelInfo.grid(problem), 
                                        kernelInfo.block(), kernelInfo.sharedMemSize(problem), stream);
  status = cudaGetLastError();
  CUDA_CHECK(status);

  return fastKronSuccess;
}

fastKronError CUDAKernelDatabase::invokeKernel(KernelInfo* kernel, const uint kronIndex, 
                                               KMMProblem problem, EpilogueParams epilogueParams,
                                               KernelMode execMode) {
  DistributedParams distParams;
  cudaStream_t stream = *(cudaStream_t*)streams[0];
  CUDAKernel& cudaKernel = dynamic_cast<CUDAKernel&>(*kernel);

  switch(problem.n()) {
    case 1:
      return invoke<1>(cudaKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 2:
      return invoke<2>(cudaKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 3:
      return invoke<3>(cudaKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 4:
      return invoke<4>(cudaKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 5:
      return invoke<5>(cudaKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 6:
      return invoke<6>(cudaKernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    default:
      Logger(LogLevel::Debug) << "Invalid number of fused kernels" << std::endl;
      return fastKronKernelNotFound;
  }
}

fastKronError CUDAKernelDatabase::invokeP2PStoreKernel(KernelInfo* kernel, const uint kronIndex, 
                                                       KMMProblem problem, DistributedParams distParams, 
                                                       EpilogueParams epilogueParams,
                                                       KernelMode execMode) {
  cudaStream_t stream = *(cudaStream_t*)streams[distParams.proc()];
  CUDAKernel& cudaKernel = dynamic_cast<CUDAKernel&>(*kernel);

  switch (problem.n()) {
    case 1:
      return invoke<1>(cudaKernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    case 2:
      return invoke<2>(cudaKernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    case 3:
      return invoke<3>(cudaKernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    case 4:
      return invoke<4>(cudaKernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    case 5:
      return invoke<5>(cudaKernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    case 6:
      return invoke<6>(cudaKernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    default:
      Logger(LogLevel::Debug) << "Invalid number of fused kernels" << std::endl;
  }

  return fastKronKernelNotFound;
}

fastKronError CUDAKernelDatabase::timeKernel(KernelInfo* kernel, const uint factorIdx, 
                                           KMMProblem problem, DistributedParams distParams, 
                                           EpilogueParams epilogueParams,
                                           KernelMode execMode, 
                                           bool distP2PStore,
                                           int warmups, int runs,
                                           float& runtime) {
  // if ((dynamic_cast<CUDAKernel*>(kernel))->localSize() > 0 || 
  //     kernel->tileF.q() <= 32) {
  //   //skip probably slow kernels
  //   runtime = std::numeric_limits<float>::max();
  //   return fastKronSuccess;
  // }

  cudaStream_t stream = *(cudaStream_t*)streams[0];
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t startEvent, endEvent;
  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&endEvent));
  fastKronError status;
  for (int r = 0; r < warmups + runs; r++) {
    if (r == warmups) CUDA_CHECK(cudaEventRecord(startEvent, stream));
    if (distP2PStore) {
      status = invokeP2PStoreKernel(kernel, factorIdx, problem,
                                    distParams, epilogueParams, execMode);
    } else {
      status = invokeKernel(kernel, factorIdx, problem,
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

static float blocksPerSM(const CUDAArchDetails gpu, CUDAKernel* kernel, dim3 grid) {
  uint32_t regOcc = gpu.regsPerSM / (kernel->block().x * kernel->numRegs());
  uint32_t shmemOcc = gpu.sharedMemPerSM / kernel->sharedMemSize();
  return min(min(regOcc, shmemOcc), gpu.maxBlocksPerSM);
}

std::string CUDAKernelDatabase::occupancyDetails(KernelInfo* kernelInfo, KMMProblem problem) {
  CUDAKernel* cudaKernel = dynamic_cast<CUDAKernel*>(kernelInfo);
  std::stringstream ss;
  dim3 grid = cudaKernel->grid(problem);
  dim3 block = cudaKernel->block();
  std::string indent = "  ";

  ss << indent << "Grid          : {" << grid.x << ", " << grid.y << ", " << grid.z << "}" << std::endl
     << indent << "Block         : {" << block.x << ", " << block.y << ", " << block.z << "}" << std::endl
     << indent << "Shared Mem    : " << cudaKernel->sharedMemSize() << std::endl 
     << indent << "Reg per Thread: " << cudaKernel->numRegs() << std::endl
     << indent << "Blocks Per SM : " << blocksPerSM(getCUDADeviceProperties(), cudaKernel, cudaKernel->grid(problem)) << std::endl
     << indent << "Local Memory  : " << cudaKernel->localSize() << std::endl;

  return ss.str();
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

fastKronError CUDAKernelDatabase::procMemset(uint32_t proc, Matrix& m, float val) {
  //TODO: call a CUDA kernel for memset
  CUDA_CHECK(cudaSetDevice(proc));
  float* host = new float[m.numel()];
  memset<float>(host, m.numel(), val);
  CUDA_CHECK(cudaMemcpy(m.data(), host, m.numel()*sizeof(float), cudaMemcpyHostToDevice));
  delete host;
  return fastKronSuccess;
}

KernelInfo* CUDAKernelDatabase::kernelForSubProblem(KMMProblem subProblem, const std::vector<KernelInfo*>& kernelsForOptLevel) {
  if (kernelsForOptLevel.size() > 0) {
    //Find kernels that have either same P or same Q
    std::vector<KernelInfo*> kernelsWithSamePOrQ;
    std::copy_if(kernelsForOptLevel.begin(), kernelsForOptLevel.end(), std::back_inserter(kernelsWithSamePOrQ),
                 [subProblem](auto& kernel){return kernel->f.p() == subProblem.f(0).p() or kernel->f.q() == subProblem.f(0).q();});
    std::vector<KernelInfo*> filteredKernels;
    if (kernelsWithSamePOrQ.size() > 0) {
      filteredKernels = kernelsWithSamePOrQ;
    } else {
      filteredKernels = kernelsForOptLevel;
    }
    //sort kernels in descending order based on the number of thread blocks a kernel invoke
    auto order = [subProblem, this](auto k1, auto k2) {
      return ((CUDAKernel*)k1)->numBlocks(subProblem) > ((CUDAKernel*)k2)->numBlocks(subProblem);
    };
    std::sort(filteredKernels.begin(), filteredKernels.end(), order);
    for (auto k : filteredKernels) {
      uint blocksm = blocksPerSM(getCUDADeviceProperties(), (CUDAKernel*)k, ((CUDAKernel*)k)->grid(subProblem));
      if (((CUDAKernel*)k)->numBlocks(subProblem) <= getCUDADeviceProperties().numSMs * blocksm) {
        return k;
      }
    }

    //If no kernel is found then return the kernel with max reuse
    return filteredKernels[filteredKernels.size() - 1];
  }

  return nullptr;
}

std::map<uint32_t, std::vector<KernelInfo*>, std::greater<int>> CUDAKernelDatabase::filterFastestFusedKernels(const KMMProblem& problem, const std::vector<KernelInfo*>& kernels) {
  uint32_t MinConsecutiveStoreElems = (getCUDADeviceProperties().smArch == SMArch::ampere) ? 16 : 8; //TODO: 16 for Ampere and 8 for Volta
  //A fused kernel stores logP (TK) consecutive elements.
  //Remove all kernels that stores (< MinConsecutiveStoreElems).
  std::vector<KernelInfo*> validFusedKernels;
  
  {
    auto filter = [problem, MinConsecutiveStoreElems](KernelInfo* kernel) {
      const int PpowerN = (int)powf(problem.f(0).p(), kernel->FusedFacs);
      const int consecutiveStoreElems = kernel->tileX.n()/PpowerN;
      return consecutiveStoreElems >= MinConsecutiveStoreElems;
    };

    std::copy_if(kernels.begin(), kernels.end(), std::back_inserter(validFusedKernels), filter);
  }
  
  if (false) {
    for (auto iter : validFusedKernels) {
      std::cout << iter->str() << std::endl;
    }
  }

  return KernelDatabase::filterFastestFusedKernels(problem, validFusedKernels);
}

int CUDAKernelDatabase::numDevices() {
  int devs;
  CUDA_CHECK(cudaGetDeviceCount(&devs));
  return devs;
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

fastKronError CUDAKernelDatabase::init(void* ptrToStream, int gpus, int gpusInM, int gpusInK, int gpuKrons) {
  numGPUs_ = gpus;
  setCUDAStream(ptrToStream);
  if (numGPUs_ > numDevices()) return fastKronInvalidArgument;
  isDistributed_ = gpus > 1;

  for (int i = 0; i < numGPUs_; i++) {
    auto detail = new CUDAArchDetails(i);
    hardware.push_back(detail);
    Logger(LogLevel::Info) << "Detected GPU " << i << std::endl << (*detail);
  }

  if (isDistributed_) {
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