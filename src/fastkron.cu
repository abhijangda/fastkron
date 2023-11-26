#include <cassert>

#include <iostream>
#include <unordered_map>

#include "handle.h"
#include "device/params.h"
#include "env.h"
#include "device/kernel_info.h"
#include "kernel_defs.cuh"
#include "autotuner.h"

/**************************************************
          Library Functions
***************************************************/
void FastKronHandle_init(FastKronHandle& handle, int gpus, int gpusInM, int gpusInK, int gpuKrons) {
  //TODO: Support both modes. Single Process multi gpu and multi process multi gpu
  handle.isDistributed_ = gpus > 1;
  if (handle.isDistributed_) {
    //TODO: Setting DistComm in another function
    handle.setUseFusion(false);
    handle.numGPUs_ = gpus;
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

    handle.distComm_ = env::getDistComm();

    if (handle.distComm_ == DistComm::P2P) {
      if (!allP2PAccess) {
        std::cout << "P2P Access among GPUs not available using NCCL" << std::endl;
        handle.distComm_ = DistComm::DistCommNone;
      }
    } else if (handle.distComm_ == DistComm::NCCL) {
      int devs[gpus];
      handle.distComm_ = DistComm::NCCL;
      ncclUniqueId ncclId;
      ncclGetUniqueId(&ncclId);
      std::cout << "Initializing NCCL"<<std::endl;
      for (int i = 0; i < gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        handle.ncclComms.push_back(nullptr);
        devs[i] = i;
      }
      NCCLCHECK(ncclCommInitAll(&handle.ncclComms[0], gpus, devs));
    }

    if (handle.distComm_ == DistComm::DistCommNone) {
      if (allP2PAccess) {
        handle.distComm_ = DistComm::P2P;
      } else {
        int devs[gpus];
        handle.distComm_ = DistComm::NCCL;
        ncclUniqueId ncclId;
        ncclGetUniqueId(&ncclId);
        std::cout << "Initializing NCCL"<<std::endl;
        for (int i = 0; i < gpus; i++) {
          CUDA_CHECK(cudaSetDevice(i));
          handle.ncclComms.push_back(nullptr);
          devs[i] = i;
        }
        NCCLCHECK(ncclCommInitAll(&handle.ncclComms[0], gpus, devs));
      }
    }

    std::cout << "Using " << handle.distComm_ << " for distributed comm" << std::endl;

    if (gpusInK >= 1)
      handle.gpusInK_ = gpusInK;
    else
      handle.gpusInK_ = 2;//ilog2(gpus);
    
    if (gpusInM >= 1)
      handle.gpusInM_ = gpusInM;  
    else
      handle.gpusInM_ = 1;//ilog2(gpus);
      
    //TODO: Check that gpuKrons batch is valid, i.e., P1*P2..PBatch <= gpusInK
    if (gpuKrons > 0)
      handle.perGPUKronBatch_ = gpuKrons;
    else 
      handle.perGPUKronBatch_ = 1;

    //TODO: Check if gpusInK_ == 1 then perGPUKronBatch = NumKrons

    std::cout << "gpusInRows " << handle.gpusInM_ <<
                 " gpusInCols " << handle.gpusInK_ << 
                 " gpuKronBatch " << handle.perGPUKronBatch_ <<
                 std::endl;
    if (handle.gpusInK_ * handle.gpusInM_ != handle.numGPUs_)  {
      std::cout << "gpusInCols * gpusInRows != total gpus (" << 
                   handle.gpusInK_ * handle.gpusInM_ << "!= " << 
                   handle.numGPUs_<< ")" << std::endl;
      abort();
    }
    //TODO: Check that localKrons <= log (gpuK_)_P
    // handle.gpuM_ = handle.M_/handle.gpusInM_;
    // handle.gpuK_ = handle.K_/handle.gpusInK_;
    // handle.gpuN_ = handle.N_/handle.gpusInK_;
    
    //All gpus with same row shares the same barrier
    //TODO: free
    handle.barriers_ = new pthread_barrier_t[handle.gpusInM_];
    handle.threads_ = new thread_pool<ThreadArgs*>(handle.numGPUs_);

    for (int i = 0; i < handle.gpusInM_; i++) {
      int s = pthread_barrier_init(&handle.barriers_[i], NULL, handle.gpusInK_);
      //TODO: Create PTHREAD_CHECK?
      assert (s == 0);
    }
    
    // size_t tempN = handle.gpuK_;
    // size_t maxTempN = tempN;
    // for (int i = 0; i < handle.NumKronMats_; i++) {
    //   tempN = (tempN/handle.KronMatRows_[i])*handle.KronMatCols_[i];
    //   if (maxTempN < tempN)
    //     maxTempN = tempN;
    // }

    // size_t sz = handle.gpuM_ * maxTempN * sizeof(T);
    // std::cout << "Allocating temporaries of size "<< sz << std::endl;
    // std::cout << "Allocated temporaries"<<std::endl;

  }

  //Load kernels into compiledKernels map
  for (uint i = 0; i < sizeof(KronGemmKernels)/sizeof(KernelInfo); i++) {
    KernelInfo& info = KronGemmKernels[i];
    KronMatmulShape shape {info.KronCols, info.KronRows, info.MaxColsA, 0, info.NumFusedKerns, info.DistributeToGPUs};
    auto iter = handle.compiledKernels.find(shape);
    if (iter == handle.compiledKernels.end()) {
      handle.compiledKernels.emplace(std::make_pair(shape, std::vector<KernelInfo>()));
    }
    handle.compiledKernels.at(shape).push_back(info);
  }
  
  //TODO: Check that if distP2PStore is needed then there is a kernel that can 
  //do it
  //TODO: Add if debug
  if (false) {
    uint numKernels = 0;
    std::cout << "Loading compiled kernels" << std::endl;
    for (auto iter : handle.compiledKernels) {
      for (auto kernel : iter.second) {
        // std::cout << kernel << std::endl;
      }
      numKernels += iter.second.size();
    }
    std::cout << "Number of kernels loaded: " << numKernels << std::endl;
  }
}

void FastKronHandle::free() {
  if (isDistributed_) {
    //TODO: Clear everything
    for (uint g = 0; g < gpusInM_; g++) {
      int s = pthread_barrier_destroy(&barriers_[g]);
      assert (s == 0);
    }

    delete threads_;
    delete barriers_;

    if (distComm_ == DistComm::NCCL) {
      for (int i=0; i<ncclComms.size(); i++)
        ncclCommDestroy(ncclComms[i]);
    }

    // delete[] gpuTemp1_;
    // delete[] gpuTemp2_;

    // gpuTemp1_ = nullptr;
    // gpuTemp2_ = nullptr;
  } else {
    // CUDA_CHECK(cudaFree(temp1_));
    // CUDA_CHECK(cudaFree(temp2_));
  
    // temp1_ = nullptr;
    // temp2_ = nullptr;  
  }
  compiledKernels.clear();
}

cudaError_t fastKronInit(fastKronHandle* handle, int gpus, int gpusInM, int gpusInK, int gpuLocalKrons) {
  FastKronHandle* h = new FastKronHandle;
  FastKronHandle_init(*h, gpus, gpusInM, gpusInK, gpuLocalKrons);
  *handle = h;
  return cudaSuccess;
}

void fastKronDestroy(fastKronHandle handle) {
  handle->free();
  delete handle;
}

cudaError_t sgekmm(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], float* result,
                   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float* temp1, float* temp2,
                   float alpha, float beta, float *z, cudaStream_t stream) {
  return handle->sgekmm(NumKronMats, x, kronMats, result,
                                            M, N, K, KronMatCols, KronMatRows, temp1, temp2, 
                                            EpilogueParams<float>(alpha, beta, z), stream);
}

cudaError_t igekmm(fastKronHandle handle, const uint NumKronMats, int* x, int* kronMats[], int* result,
                   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], int* temp1, int* temp2,
                   int alpha, int beta, int *z, cudaStream_t stream) {
  return handle->igekmm(NumKronMats, x, kronMats, result, 
                                        M, N, K, KronMatCols, KronMatRows, temp1, temp2,
                                        EpilogueParams<int>(alpha, beta, z), stream);
}

cudaError_t dgekmm(fastKronHandle handle, const uint NumKronMats, double* x, double* kronMats[], double* result,
                   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], double* temp1, double* temp2,
                   double alpha, double beta, double *z, cudaStream_t stream) {
  return cudaSuccess;
                    // return handle->gekmm(FastKronType::Double, NumKronMats, x, kronMats, result, 
  //                                             M, N, K, KronMatCols, KronMatRows, temp1, temp2,
  //                                             EpilogueParams<double>(alpha, beta, z), stream);
}


cudaError_t kronSGEMMOutofCore(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream) {
  // return singleGPUOutOfCoreKronMatmul<float, float4>(handle, NumKronMats, x, kronMats, result, 
  //                                                    M, N, K, KronMatCols, KronMatRows, stream);
}

// cudaError_t kronSGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
//   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]) {
//   return singleGPUOutOfCoreKronMatmul<float, float4>(handle, NumKronMats, x, kronMats, result, 
//                                                      M, N, K, KronMatCols, KronMatRows, stream);
// }

// cudaError_t kronIGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], int** result,
//   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]) {
//   return singleGPUOutOfCoreKronMatmul<int, int4>(handle, NumKronMats, x, kronMats, result, 
//                                                  M, N, K, KronMatCols, KronMatRows, stream);
// }

cudaError_t kronDistributedSGEMM(fastKronHandle handle, const uint NumKronMats, float* x[], float* kronMats[], float* result[],
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
                                 cudaStream_t streams[]) {
  return handle->distributedsgekmm(NumKronMats, x, kronMats, result, M, N, K, 
                                   KronMatCols, KronMatRows, temp1, temp2, streams);
}

cudaError_t gekmmSizes(fastKronHandle handlePtr, const uint NumKronMats, uint M, uint N, uint K, 
                          uint KronMatCols[], uint KronMatRows[], size_t* resultSize, size_t* tempSize) {
  if (resultSize == nullptr) return cudaErrorInvalidValue;
  if (tempSize   == nullptr) return cudaErrorInvalidValue;
  uint gpuM, gpuK;
  FastKronHandle& handle = *handlePtr;
  if (handle.isDistributed_) {
    if (!checkDistributedKronSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows, 
                                   handle.perGPUKronBatch_, handle.gpusInK_))
      return cudaErrorInvalidValue;
    gpuM = M/handle.gpusInM_;
    gpuK = K/handle.gpusInK_;
  } else {
    if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
      return cudaErrorInvalidValue;
    gpuM = M;
    gpuK = K;
  }
  size_t tempN = gpuK;
  size_t maxTempN = tempN;
  for (int i = NumKronMats - 1; i >= 0; i--) {
    tempN = (tempN/KronMatRows[i])*KronMatCols[i];
    if (maxTempN < tempN)
      maxTempN = tempN;
  }

  *tempSize   = gpuM * maxTempN;
  if (handle.isDistributed_ and handle.distComm_ == DistComm::NCCL)
    //Include size of send and recv buffers 
    *tempSize = (*tempSize) * 2;
  *resultSize = gpuM * tempN;

  return cudaSuccess;
}

cudaError_t sgekmmTune(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], 
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                                 cudaStream_t stream) {
  return Autotuner().tune(*handle, NumKronMats, x, kronMats,
                         M, N, K, KronMatCols, KronMatRows,
                         stream);
}

cudaError_t dgekmmTune(fastKronHandle handle, const uint NumKronMats, double* x, double* kronMats[], 
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream) {
  return Autotuner().tune(*handle, NumKronMats, x, kronMats, 
                          M, N, K, KronMatCols, KronMatRows,
                          stream);
}

cudaError_t idgemmTune(fastKronHandle handle, const uint NumKronMats, int* x, int* kronMats[],
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream) {
  return Autotuner().tune(*handle, NumKronMats, x, kronMats,
                       M, N, K, KronMatCols, KronMatRows,
                       stream);
}

void FastKronHandle::getDistributedSizes(uint M, uint K, uint& gpuM, uint& gpuK) {
  gpuM = M/gpusInM_;
  gpuK = K/gpusInK_;
}