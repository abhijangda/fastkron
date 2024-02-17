#include <iostream>
#include <iomanip>

#include "utils/utils.h"
#include "kernel_db/cuda_kernel_db.h"
#include "kernel_db/kernel_defs.h"

static bool isValidKernel(KernelInfo& kernelInfo) {
  const uint NumThreads = kernelInfo.NumThreads;
  const uint CRegRows = kernelInfo.CRegRows;
  const uint CRegCols = kernelInfo.CRegCols;
  const Factor& tiledFactor = kernelInfo.tiledFactor;

  const uint ValidThreads = ((kernelInfo.tiledInput.n()/kernelInfo.factor.p())/CRegRows) * (tiledFactor.q()/CRegCols);
  if (NumThreads != ROUNDUP(ValidThreads, CUDA_WARP_SIZE)) {
    std::cout << "Invalid kernel config " << kernelInfo << std::endl; 
    return false;
  }

  return true;
}

CUDAKernelDatabase::CUDAKernelDatabase() {
  streams.push_back(NULL);

  //Load kernels into compiledKernels map
  for (uint i = 0; i < sizeof(CUDAKernels)/sizeof(KernelInfo); i++) {
    KernelInfo& info = CUDAKernels[i];
    if (!isValidKernel(info)) abort();
    CUDA_CHECK(info.setSharedMemAttr());
    //  {info.KronCols, info.KronRows, info.MaxColsA, 0, info.NumFusedKerns, info.DistributeToGPUs};
    DbKey key {info.factor, info.opX, info.opF};
    auto iter = compiledKernels.find(key);
    if (iter == compiledKernels.end()) {
      compiledKernels.emplace(std::make_pair(key, std::vector<KernelInfo>()));
    }
    compiledKernels.at(key).push_back(info);
  }
  
  //TODO: Check that if distP2PStore is needed then there is a kernel that can 
  //do it
  //TODO: Add if debug
  if (false) {
    uint numKernels = 0;
    std::cout << "Loading compiled kernels" << std::endl;
    for (auto iter : compiledKernels) {
      for (auto kernel : iter.second) {
        // std::cout << kernel << std::endl;
      }
      numKernels += iter.second.size();
    }
    std::cout << "Number of kernels loaded: " << numKernels << std::endl;
  }
}

//Launch cuda kernels
template<uint NumFusedKerns>
cudaError_t invoke(KernelInfo& kernelInfo, const uint kronIndex, 
                   KMMProblem problem,
                   DistributedParams distParams,
                   EpilogueParams epilogueParams,
                   KernelMode execMode,
                   cudaStream_t stream) {
  cudaError_t status;

  //Create the grid and thread block
  KernelParams<NumFusedKerns> params (problem, kronIndex, execMode);
  FusedParams<NumFusedKerns> fusedParams (problem, kernelInfo.tiledInput.n());

  //Call kernel
  typedef void (*KronMatmulKernelTy)(KernelParams<NumFusedKerns>, FusedParams<NumFusedKerns>, 
                                     DistributedParams, EpilogueParams, dim3, dim3, uint32_t, cudaStream_t);
  KronMatmulKernelTy(kernelInfo.invokerFunc)(params, fusedParams, distParams, 
                                        epilogueParams, kernelInfo.grid(problem), 
                                        kernelInfo.block(), kernelInfo.sharedMemSize(), stream);
  status = cudaGetLastError();
  CUDA_CHECK(status);
  return status;
}

cudaError_t CUDAKernelDatabase::invokeKernel(KernelInfo& kernel, const uint kronIndex, 
                                         KMMProblem problem, EpilogueParams epilogueParams,
                                         KernelMode execMode) {
  DistributedParams distParams;
  cudaStream_t stream = streams[0];
  switch(problem.n()) {
    case 1:
      return invoke<1>(kernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 2:
      return invoke<2>(kernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 3:
      return invoke<3>(kernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 4:
      return invoke<4>(kernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 5:
      return invoke<5>(kernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 6:
      return invoke<6>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    default:
      std::cout << "Invalid number of fused kernels" << std::endl;
      return cudaErrorInvalidValue;
  }
}

cudaError_t CUDAKernelDatabase::invokeP2PStoreKernel(KernelInfo& kernel, const uint kronIndex, 
                                                 KMMProblem problem, DistributedParams distParams, 
                                                 EpilogueParams epilogueParams,
                                                 KernelMode execMode) {
  cudaStream_t stream = streams[distParams.proc()];
  switch (problem.n()) {
    case 1:
      return invoke<1>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    case 2:
      return invoke<2>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    case 3:
      return invoke<3>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    case 4:
      return invoke<4>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    case 5:
      return invoke<5>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    case 6:
      return invoke<6>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    default:
      std::cout << "Invalid number of fused kernels" << std::endl;
  }

  return cudaErrorInvalidValue;
}

std::pair<KernelInfo, float> CUDAKernelDatabase::tuneKernelForProblem(KMMProblem problem, bool distP2PStore, 
    uint factorIdx, DistributedParams distParams) {
  const uint runs = 5;
  const uint warmups = 2;
  KernelInfo bestKernel;
  cudaEvent_t start, end;
  float minTime;
  bool foundProblem = false;
  std::vector<KernelInfo> allKernels;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  cudaStream_t stream = streams[0];
  minTime = std::numeric_limits<float>::max();

  if (findAllKernels(problem.f(0), problem.opX(), problem.opFs(), allKernels)) {
  for (auto kernel : allKernels) {
    if (!kernel.canCompute(problem, distP2PStore)) continue;
    if (!foundProblem) {
      std::cout << "Tuning for shape "  << problem << std::endl;
      foundProblem = true;
    }
    std::cout << kernel;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaError_t status;
    for (int r = 0; r < warmups + runs; r++) {
      if (r == warmups) CUDA_CHECK(cudaEventRecord(start, stream));
      if (distP2PStore) {
        status = invokeP2PStoreKernel(kernel, factorIdx, problem,
                                      distParams, EpilogueParams::create<float>(), KernelModeTuning);
      } else {
        status = invokeKernel(kernel, factorIdx, problem,
                              EpilogueParams::create<float>(), KernelModeTuning);
      }
    }
    CUDA_CHECK(status);
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    
    if (status != cudaSuccess)
      std::cout << "Error: " << cudaGetErrorString(status) << std::endl;
    float kernelTime;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, end));
    std::cout << std::fixed << std::setprecision(2) << 
                " runs in " << (kernelTime/runs) << " ms " << std::endl;
    if (kernelTime < minTime) {
      bestKernel = kernel;
      minTime = kernelTime;
    }
  }}

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  if (minTime < std::numeric_limits<float>::max()) {
    std::cout << std::fixed << std::setprecision(2) <<
                "Best kernel for " << problem << ": " << bestKernel << " runs in " << (minTime/runs) << " ms" << std::endl;
    return std::make_pair(bestKernel, minTime/runs);
  }

  return std::make_pair(bestKernel, minTime);
}

cudaError_t CUDAKernelDatabase::procMalloc(uint32_t proc, size_t size, void*& ptr) {
  CUDA_CHECK(cudaSetDevice(proc));
  CUDA_CHECK(cudaMalloc(&ptr, size));
  CUDA_CHECK(cudaMemset(ptr, 1, size));
  
  return cudaSuccess;
}

cudaError_t CUDAKernelDatabase::procMalloc(uint32_t proc, Matrix& m) {
  void* ptr = nullptr;
  cudaError_t e = procMalloc(proc, m.numel() * sizeof(float), ptr);

  if (e == cudaSuccess) {
    m.ptr = ptr;
  }

  return e;
}

cudaError_t CUDAKernelDatabase::procFree(uint32_t proc, void* ptr) {
  CUDA_CHECK(cudaSetDevice(proc));
  CUDA_CHECK(cudaFree(ptr));
  return cudaSuccess;
}

cudaError_t CUDAKernelDatabase::procFree(uint32_t proc, Matrix m) {
  return procFree(proc, m.data());
}

cudaError_t CUDAKernelDatabase::procMemset(uint32_t proc, Matrix& m, float val) {
  //TODO: call a CUDA kernel for memset
  CUDA_CHECK(cudaSetDevice(proc));
  float* host = new float[m.numel()];
  for (int i = 0; i < m.numel(); i++)
    host[i] = val;
  CUDA_CHECK(cudaMemcpy(m.data(), host, m.numel()*sizeof(float), cudaMemcpyHostToDevice));
  delete host;
  return cudaSuccess;
}

cudaError_t CUDAKernelDatabase::init(void* ptrToStream, int gpus, int gpusInM, int gpusInK, int gpuKrons) {
  streams.clear();
  for (int i = 0; i < gpus; i++) {
    streams.push_back(((cudaStream_t*)ptrToStream)[i]);
  }
  numGPUs_ = gpus;
  isDistributed_ = gpus > 1;
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
        std::cout << "P2P Access among GPUs not available using NCCL" << std::endl;
        distComm_ = DistComm::DistCommNone;
      }
    } else if (distComm_ == DistComm::NCCL) {
      int devs[gpus];
      distComm_ = DistComm::NCCL;
      ncclUniqueId ncclId;
      ncclGetUniqueId(&ncclId);
      std::cout << "Initializing NCCL"<<std::endl;
      for (int i = 0; i < gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        ncclComms.push_back(nullptr);
        devs[i] = i;
      }
      NCCLCHECK(ncclCommInitAll(&ncclComms[0], gpus, devs));
    }

    if (distComm_ == DistComm::DistCommNone) {
      if (allP2PAccess) {
        distComm_ = DistComm::P2P;
      } else {
        int devs[gpus];
        distComm_ = DistComm::NCCL;
        ncclUniqueId ncclId;
        ncclGetUniqueId(&ncclId);
        std::cout << "Initializing NCCL"<<std::endl;
        for (int i = 0; i < gpus; i++) {
          CUDA_CHECK(cudaSetDevice(i));
          ncclComms.push_back(nullptr);
          devs[i] = i;
        }
        NCCLCHECK(ncclCommInitAll(&ncclComms[0], gpus, devs));
      }
    }

    std::cout << "Using " << distComm_ << " for distributed comm" << std::endl;

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

    std::cout << "gpusInRows " << gpusInM_ <<
                 " gpusInCols " << gpusInK_ << 
                 " gpuKronBatch " << perGPUKronBatch_ <<
                 std::endl;
    if (gpusInK_ * gpusInM_ != numGPUs_)  {
      std::cout << "gpusInCols * gpusInRows != total gpus (" << 
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

  return cudaSuccess;
}