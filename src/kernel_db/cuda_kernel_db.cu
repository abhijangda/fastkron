#include <iostream>
#include <iomanip>

#include <nccl.h>

#include "utils/utils.h"
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

CUDAKernelDatabase::CUDAKernelDatabase() {
  streams.push_back(NULL);
  loadKernels<CUDAKernel>(AllCUDAKernels, sizeof(AllCUDAKernels)/sizeof(CUDAKernel));
  for (uint i = 0; i < sizeof(AllCUDAKernels)/sizeof(CUDAKernel); i++) {
    CUDAKernel& info = AllCUDAKernels[i];
    if (!info.isValid()) abort();
    CUDA_CHECK(info.setSharedMemAttr());
  }
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
  KernelParams<FusedFacs> params (problem, kernelInfo.getTileX(problem), 
                                  kernelInfo.getTileF(problem), 
                                  kronIndex, execMode);
  FusedParams<FusedFacs> fusedParams (problem, kernelInfo.tileX.n());

  std::cout << "72: " << kernelInfo.grid(problem).x << " " << kernelInfo.grid(problem).y << std::endl;
  std::cout << "73: " << kernelInfo.getTileX(problem) << std::endl;
  std::cout << "74: " << kernelInfo.getTileF(problem) << std::endl;
  std::cout << "75: " << kernelInfo.sharedMemSize(problem) << std::endl;
  //Call kernel
  typedef void (*KronMatmulKernelTy)(KernelParams<FusedFacs>, FusedParams<FusedFacs>, 
                                     DistributedParams, EpilogueParams, dim3, dim3, uint32_t, cudaStream_t);
  KronMatmulKernelTy(kernelInfo.invokerFunc)(params, fusedParams, distParams, 
                                        epilogueParams, kernelInfo.grid(problem), 
                                        kernelInfo.block(), kernelInfo.sharedMemSize(problem), stream);
  status = cudaGetLastError();
  CUDA_CHECK(status);

  if (false && kronIndex == 1) {
    printf("80\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    float* m = new float[problem.x().numel()];
    cudaMemcpy(m, params.problem.y().data(), params.problem.y().numel() * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < problem.y().numel(); i++) {
      if (m[i] != 31) {
        printf("%f %d %d\n", m[i], i/(problem.y().n()), i%(problem.y().n()));
        break;
      }
    }
    exit(EXIT_SUCCESS);
  }

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
      std::cout << "Invalid number of fused kernels" << std::endl;
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
      std::cout << "Invalid number of fused kernels" << std::endl;
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
    std::cout << "Error: " << fastKronGetErrorString(status) << std::endl;
    return status;
  }
  CUDA_CHECK(cudaEventElapsedTime(&runtime, startEvent, endEvent));
  runtime = runtime/runs;
  CUDA_CHECK(cudaEventDestroy(startEvent));
  CUDA_CHECK(cudaEventDestroy(endEvent));
  return status;
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

fastKronError CUDAKernelDatabase::init(void* ptrToStream, int gpus, int gpusInM, int gpusInK, int gpuKrons) {
  streams.clear();
  cudaStream_t* t = new cudaStream_t;
  *t = 0;
  for (int i = 0; i < gpus; i++) {
    if (ptrToStream != NULL)
	  streams.push_back(((cudaStream_t*)ptrToStream) + i);
    else
	    streams.push_back(t);
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
        std::cout << "Initializing NCCL"<<std::endl;
        for (int i = 0; i < gpus; i++) {
          CUDA_CHECK(cudaSetDevice(i));
          ncclComms.push_back(nullptr);
          devs[i] = i;
        }
        NCCLCHECK(ncclCommInitAll((ncclComm_t*)&ncclComms[0], gpus, devs));
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

  return fastKronSuccess;
}
