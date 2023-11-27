#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <type_traits>
#include <thread>

#include <vector>
#include <iostream>
#include <sstream>
#include <limits>
#include <iomanip>
#include <cstring>

#include "utils.h"
#include "handle.h"
#include "thread_pool.h"
#include "device/otherkernels.cuh"
#include "env.h"
#include "autotuner.h"
#include "kernel_defs.cuh"

/*TODOs:
 1. Using fusion or not should be an environemnt flag
 2. Debug message environment flag*/

std::size_t std::hash<KronMatmulShape>::operator()(const KronMatmulShape& k) const {
  return hash<uint>()(k.KronCols) ^ hash<uint>()(k.KronRows) ^ hash<uint>()(k.ColsA);
}

/**Library entry points to launch cuda kernels**/

//Check N and K is a multiplication of KronMatCols and KronMatRows
bool checkKronMatrixSizes(const uint NumKronMats, 
                                 const uint M, const uint N, const uint K, 
                                 const uint KronMatCols[], const uint KronMatRows[]) {
  uint n=1,k=1;
  for (uint i = 0; i < NumKronMats; i++) {
    k *= KronMatRows[i];
    n *= KronMatCols[i];
  }
  if (n != N || k != K) {
    printf("Invalid Kron product sizes %d != %d, %d != %d\n", n, N, k, K);
    return false;
  }

  return true;
}

bool checkDistributedKronSizes(const uint NumKronMats, 
                                      const uint M, const uint N, const uint K, 
                                      const uint KronMatCols[], const uint KronMatRows[],
                                      const uint LocalKrons, const uint gpusInK) {
  uint prevTempN = K;
  
  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return false;
  
  if (prevTempN % gpusInK != 0) return false;
    
  for (uint i = 0; i < NumKronMats; i += LocalKrons) {
    const uint kronMat = NumKronMats - i - 1;
    const uint NumFusedKerns = min(LocalKrons, NumKronMats - i);
    uint currTempN = prevTempN;
    // printf("243: NumFusedKerns %d kronMat \n", NumFusedKerns);
    uint FusedKronMatCols[NumFusedKerns];
    uint FusedKronMatRows[NumFusedKerns];
    for (int k = 0; k < NumFusedKerns; k++) {
      FusedKronMatCols[k] = KronMatCols[kronMat - k];
      FusedKronMatRows[k] = KronMatRows[kronMat - k];
      currTempN = (currTempN/FusedKronMatRows[k])*FusedKronMatCols[k];
    }
  
    if (currTempN % gpusInK != 0) return false;
    prevTempN = currTempN;
  }

  return true;
}

KronMatmulShape FastKronHandle::maxCompiledColsA(KronMatmulShape shape) {
  while (compiledKernels.find(shape) == compiledKernels.end()) {
    shape.ColsA /= 2;
    if (shape.ColsA == 1) {
     break;
    }
  }

  return shape;
}

uint FastKronHandle::maxFusedKernels(KronMatmulShape shape) {
  uint numFusedKernels = 0;
  //Go through fused kernels starting from 1 
  //find if the shape exists for the fused kernel
  //if it exist then go to next fused kernel
  while (true) {
    shape.NumFusedKerns = numFusedKernels + 1;
    auto shapeFound = maxCompiledColsA(shape);
    if (shapeFound.ColsA == 1) {
      break;
    }
    numFusedKernels++;
  }

  return numFusedKernels;
}

KernelInfo FastKronHandle::selectKernel(KronMatmulShape shape) {
  //Go through all MaxColsA starting from MAX_K and select the relevant
  KronMatmulShape maxColsAShape = maxCompiledColsA(shape);
  //TODO: Remove kEqVar. it provides only a little improvement in perf
  //but makes writing code hard
  int kEqVar = 0; //(maxColsAShape.ColsA == shape.ColsA) ? 1 : 0;
  auto iter = compiledKernels.find(maxColsAShape);
  if (iter == compiledKernels.end()) {
    std::cout << "No kernel found for " << shape << std::endl;
    abort();
    return KernelInfo{};
  }
  auto kernelInfos = iter->second;
  KernelInfo kernelInfo;
  for (auto info : kernelInfos) {
    //TODO: need to check for type
    //TODO: make use of KernelInfo.canCompute
    if (info.KEqVar == kEqVar) {
      uint tileRowA = info.TileRowsA;
      bool row_mod_tile_zero = (shape.RowsA % tileRowA) == 0;    
      if (info.RowModTileIsZero == row_mod_tile_zero) {
        return info;
      }
    }
  }

  std::cout<<"No kernel selected" << std::endl;
  abort();
  return KernelInfo();
}

//TODO: These methods that take handle should be private methods of FastKronHandle
TunedKernelsSeries selectKernelSeries(FastKronHandle& handle, const uint NumKronMats,
                                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                                      bool distributedKernel) {
  uint MaxFusedKerns = handle.getUseFusion() ? handle.maxFusedKernels(KronMatmulShape{KronMatCols[0], KronMatRows[0], K, M, 0}) : 1;
  MaxFusedKerns = min(MaxFusedKerns, NumKronMats);
  TunedKernelsSeries tunedSeries;
  uint prevTempN = K;
  for (uint i = 0; i < NumKronMats; i += MaxFusedKerns) {
    const uint kronMat = NumKronMats - i - 1;
    const uint NumFusedKerns = min(MaxFusedKerns, NumKronMats - i);
    uint currTempN = prevTempN;
    // printf("243: NumFusedKerns %d kronMat \n", NumFusedKerns);
    uint FusedKronMatCols[NumFusedKerns];
    uint FusedKronMatRows[NumFusedKerns];
    for (int k = 0; k < NumFusedKerns; k++) {
      FusedKronMatCols[k] = KronMatCols[kronMat - k];
      FusedKronMatRows[k] = KronMatRows[kronMat - k];
      currTempN = (currTempN/FusedKronMatRows[k])*FusedKronMatCols[k];
    }
  
    bool DistributeToGPUs = distributedKernel && handle.distComm_ == DistComm::P2P && handle.gpusInK_ > 1 && (i == NumKronMats - 1);
    auto selectedKernel = handle.selectKernel(KronMatmulShape{KronMatCols[kronMat], KronMatRows[kronMat], 
                                       prevTempN, M, NumFusedKerns, DistributeToGPUs});
    tunedSeries.push_back({selectedKernel, kronMat - NumFusedKerns, kronMat, prevTempN, 0.0f});
    prevTempN = currTempN;
  }

  return tunedSeries;
}

cudaError_t FastKronHandle::xgekmm(const uint NumKronMats, void* x, void** kronMats,
                                void* result,
                                uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], 
                                void* temp1, void* temp2, 
                                EpilogueParams epilogueParams,
                                cudaStream_t stream) {
  //Only row major layout of all matrics is supported.
  if (result == nullptr) return cudaErrorInvalidValue;
  if (temp1  == nullptr) return cudaErrorInvalidValue;

  if (!checkKronMatrixSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows))
    return cudaErrorInvalidValue;
  
  void* kronGemmResults[2] = {temp1, temp2};
  void* prevKronResult = x;
  void* currKronResult = kronGemmResults[0];

  //TODO: Assumes all factors are of same size and square shape
  TunedKernelsSeries kernelSeries;
  if (tunedKernelSeries.size() > 0) {
    kernelSeries = tunedKernelSeries;
  } else {
    kernelSeries = selectKernelSeries(*this, NumKronMats, M, N, K, 
                                      KronMatCols, KronMatRows, false);
  }

  if (temp2 == nullptr) {
    if (kernelSeries.size() % 2 == 1) {
      kronGemmResults[0] = result;
      kronGemmResults[1] = temp1;
    } else {
      kronGemmResults[0] = temp1;
      kronGemmResults[1] = result;
    }

    currKronResult = kronGemmResults[0];
    prevKronResult = x;
  }

  //Use double buffering for writing result and using output 
  //of previous iteration as input to current
  uint prevTempN = K;
  uint currTempN;
  for (auto kernel : kernelSeries) {
    const uint kronMat = kernel.end;
    const uint NumFusedKerns = kernel.kernel.NumFusedKerns;
    void* krons[NumFusedKerns];
    uint FusedKronMatCols[NumFusedKerns];
    uint FusedKronMatRows[NumFusedKerns];
    currTempN = prevTempN;
    for (int k = 0; k < NumFusedKerns; k++) {
      krons[k] = kronMats[kronMat - k];
      FusedKronMatCols[k] = KronMatCols[kronMat - k];
      FusedKronMatRows[k] = KronMatRows[kronMat - k];
      currTempN = (currTempN/FusedKronMatRows[k])*FusedKronMatCols[k];
    }

    //In the last iteration, write result to the results.    
    if (kronMat - NumFusedKerns + 1 == 0)
      currKronResult = result;

    cudaError_t status;

    KernelInfo selectedKernel = kernel.kernel;
    // std::cout << "Invoking " << selectedKernel << " for " << FusedKronMatCols[0] << "x" << FusedKronMatRows[0] << "  " << prevTempN << " " << currTempN << std::endl;
    status = kernelInvoker.fusedSlicedMatmul(NumFusedKerns, selectedKernel, kronMat, (void*)prevKronResult,
                               (void**)krons, (void*)currKronResult, M, currTempN, prevTempN,
                               FusedKronMatCols, FusedKronMatRows,
                               epilogueParams, stream);
    
    if (status != cudaSuccess) return status;
    
    // if (kronMat >= 1)
    // printGPUArray<float>(M, currTempN, (kronMat == 3) ? 8.0f : (kronMat == 2 ? 64.0f : 512.0f),
    //                      (float*)currKronResult, stream);
    // if (kronMat == 3) return cudaSuccess;
    prevTempN = currTempN;
    // if (kronMat == 1) return cudaSuccess;
    // return cudaSuccess;
    //Double/ring/circular buffer previous result and new result
    prevKronResult = currKronResult;
    if (prevKronResult == kronGemmResults[0]) {        
      currKronResult = kronGemmResults[1];
    } else if (prevKronResult == kronGemmResults[1]) {
      currKronResult = kronGemmResults[0];
    }
  }

  return cudaSuccess;
}

void thread_barrier_wait(pthread_barrier_t* barrier) {
  int s = pthread_barrier_wait(barrier);
  PTHREAD_BARRIER_CHECK(s);
}

void perGPUKronMatmul(ThreadArgs* thArgs) {
  // ThreadArgs<T>& thArgs = *(ThreadArgs<T>*)arg;

  FastKronHandle& handle = *thArgs->handle;
  uint NumKronMats = thArgs->NumKronMats;
  void* x = thArgs->x;
  void** kronMats = thArgs->kronMats;
  void** results = thArgs->result;
  void** temp1 = thArgs->temp1;
  void** temp2 = thArgs->temp2;
  uint M = thArgs->M;
  uint N = thArgs->N;
  uint K = thArgs->K;
  uint *KronMatCols = thArgs->KronMatCols;
  uint *KronMatRows = thArgs->KronMatRows;
  cudaStream_t* stream = thArgs->stream;
  uint gr = thArgs->gpuRow;
  uint gc = thArgs->gpuCol;
  uint gpusInM_ = thArgs->gpusInM_;
  uint gpusInK_ = thArgs->gpusInK_; 
  uint currTempN;
  uint g = gr * gpusInK_ + gc;
  CUDA_CHECK(cudaSetDevice(g));

  cudaError_t status;
  
  //Temporaries are swaped after every slicedMatmul
  //TODO: User supplied result should be used as a temp and the final results are written in it
  //TODO: What if Rows are not multiple of GPUs in Rows
  void* innerResults[2] = {temp1[g], temp2[g]};
  // std::cout << "handle.gpuM_ " << handle.gpuM_ << " handle.gpuK_ " <<handle.gpuK_ << " gpusInCols " << gpusInCols << " gpusInRows " << gpusInRows << " K " << K << std::endl;
  void* innerPrevResult;
  void* innerCurrResult;
  uint gpuM, gpuK;
  handle.getDistributedSizes(M, K, gpuM, gpuK);
  uint prevTempN = gpuK;

  uint startGpuM = gpuM * gr;
  // const uint gpuM = min(gpuM, M - startGpuM);
  //For first slicedMatmul, x is the input
  innerPrevResult = x;
  innerCurrResult = innerResults[0];
  CUDA_CHECK(cudaSetDevice(g));

  //Calculate number of swaps
  if (temp2[g] == nullptr) {
    uint currTempN;
    uint prevTempN = gpuK;
    uint numSwaps = 0;

    for (uint io = 0; io < NumKronMats; io += handle.perGPUKronBatch_) {
      uint KronMulBatchSize = min(handle.perGPUKronBatch_, NumKronMats - io);
      uint MaxI = io + KronMulBatchSize;
      const uint endKron = NumKronMats - io - KronMulBatchSize;
      
      currTempN = prevTempN;

      TunedKernelsSeries kernelSeries;
      uint LocalKronCols[KronMulBatchSize];
      uint LocalKronRows[KronMulBatchSize];
      for (int i = KronMulBatchSize - 1; i >= 0 ; i--) {
        LocalKronCols[i] = KronMatCols[NumKronMats - MaxI + i];
        LocalKronRows[i] = KronMatRows[NumKronMats - MaxI + i];
        currTempN = (currTempN/LocalKronRows[i])*LocalKronCols[i];
      }

      if (handle.tunedKernelSeries.size() > 0) {
        for (auto tunedKernel : handle.tunedKernelSeries) {
          if (tunedKernel.start >= endKron  and tunedKernel.end < endKron + KronMulBatchSize) {
            kernelSeries.insert(kernelSeries.begin(), tunedKernel);
          }
        }
      } else {
        auto localSeries = selectKernelSeries(handle, KronMulBatchSize, gpuM, gpuK, gpuK, 
                                              LocalKronCols, LocalKronRows, true);
        for (auto& kernel : localSeries) {
          kernel.end += endKron;
        }
        kernelSeries = localSeries;
      }

      numSwaps += kernelSeries.size() + ((handle.distComm_ == DistComm::P2P) ? 0 : 1);
    }

    if (numSwaps%2 == 1) {
      innerResults[0] = results[g];
      innerResults[1] = temp1[g];
    } else {
      innerResults[0] = temp1[g];
      innerResults[1] = results[g];
    }

    innerCurrResult = innerResults[0]; 
  }

  for (uint io = 0; io < NumKronMats; io += handle.perGPUKronBatch_) {
    uint KronMulBatchSize = min(handle.perGPUKronBatch_, NumKronMats - io);
    uint MaxI = io + KronMulBatchSize;
    {
      const uint endKron = NumKronMats - io - KronMulBatchSize;
      
      currTempN = prevTempN;

      TunedKernelsSeries kernelSeries;
      uint LocalKronCols[KronMulBatchSize];
      uint LocalKronRows[KronMulBatchSize];
      for (int i = KronMulBatchSize - 1; i >= 0 ; i--) {
        LocalKronCols[i] = KronMatCols[NumKronMats - MaxI + i];
        LocalKronRows[i] = KronMatRows[NumKronMats - MaxI + i];
        currTempN = (currTempN/LocalKronRows[i])*LocalKronCols[i];
      }

      if (handle.tunedKernelSeries.size() > 0) {
        for (auto tunedKernel : handle.tunedKernelSeries) {
          if (tunedKernel.start >= endKron  and tunedKernel.end < endKron + KronMulBatchSize) {
            kernelSeries.insert(kernelSeries.begin(), tunedKernel);
          }
        }
      } else {
        auto localSeries = selectKernelSeries(handle, KronMulBatchSize, gpuM, gpuK, gpuK, 
                                              LocalKronCols, LocalKronRows, true);
        for (auto& kernel : localSeries) {
          kernel.end += endKron;
        }
        kernelSeries = localSeries;
      }

      int prevFullK = prevTempN * handle.gpusInK_;
      int currFullN = currTempN * handle.gpusInK_;
      DistributedParams distParams(gr, gc, handle.gpusInK_, 
                                   prevFullK, currFullN,
                                   prevTempN, currTempN, LocalKronCols, LocalKronRows, KronMulBatchSize);
      uint slicedMuls = 0;
      bool ncclRecvInResult = false;
      for (auto kernel : kernelSeries) {
        //TODO: probably will need to change for fused kernels
        const uint NumFusedKerns = kernel.kernel.NumFusedKerns;
        
        void* krons[NumFusedKerns];
        uint kronCols[NumFusedKerns];
        uint kronRows[NumFusedKerns];
        
        currTempN = prevTempN;
        for (int kk = 0; kk < NumFusedKerns; kk++) {
          krons[kk] = kronMats[g * NumKronMats + kernel.end - kk];
          kronRows[kk] = KronMatRows[kernel.end - kk];
          kronCols[kk] = KronMatCols[kernel.end - kk];
          currTempN = (currTempN/kronRows[kk])*kronCols[kk];
        }

        if (slicedMuls == KronMulBatchSize - 1) {
          CUDA_CHECK(cudaStreamSynchronize(stream[g]));
          thread_barrier_wait(thArgs->barrier);
        }
        
        if (kernel.end - NumFusedKerns + 1 == 0) {
          if (handle.distComm_ == DistComm::P2P or handle.gpusInK_ == 1)
            innerCurrResult = results[g];
          else
            ncclRecvInResult = true;
        } 
        
        void** gpuTempResults;
        if (innerCurrResult == temp1[g]) {
          gpuTempResults = (void**)temp1;
        } else if (innerCurrResult == temp2[g]) {
          gpuTempResults = (void**)temp2;
        } else if (innerCurrResult == results[g]) {
          gpuTempResults = (void**)results;
        }
        
        void* gpuResults[handle.gpusInK_];
        for (int _gc = 0; _gc < handle.gpusInK_; _gc++) {
          gpuResults[_gc] = gpuTempResults[gr * handle.gpusInK_ + _gc];
        }
        distParams.updateGPUResults((void**)gpuResults);

        //TODO: a single switch case for FusedKernels?
        cudaError_t status;
        status = handle.kernelInvoker.fusedDistributedSlicedMatmul(NumFusedKerns, kernel.kernel, kernel.end, (void*)innerPrevResult, 
                                              (void**)krons, (void*)innerCurrResult, gpuM, currTempN, 
                                              prevTempN, kronCols, kronRows, distParams, 
                                              EpilogueParams::create<float>(), stream[g]);
        assert(status == cudaSuccess);        
        CUDA_CHECK(cudaStreamSynchronize(stream[g]));
        
        // if (gc == 0 and kernel.end == 1) {
        //   printGPUArray(handle.gpuM_, handle.gpuK_, 128.0f*128.0f, innerCurrResult, stream[g]);
        // }
        // if (gc == 0) printf("slicedMuls %d innerCurrResult %p innerPrevResult %p\n", slicedMuls, innerCurrResult, innerPrevResult);
        // if (status != cudaSuccess) goto end;
        prevTempN = currTempN;
        //Double/ring/circular buffer previous result and new result
        innerPrevResult = innerCurrResult;
        if (innerPrevResult == innerResults[0]) {
          innerCurrResult = innerResults[1];
        } else if (innerPrevResult == innerResults[1]) {
          innerCurrResult = innerResults[0];
        }
        slicedMuls++;
      }

      CUDA_CHECK(cudaStreamSynchronize(stream[g]));
      
      thread_barrier_wait(thArgs->barrier);

      if (handle.distComm_ == DistComm::NCCL && handle.gpusInK_ > 1) {
        typedef float T;
        size_t resultSize = 0, tempSize = 0;
        if (ncclRecvInResult)
          innerCurrResult = results[g];
        gekmmSizes(&handle, NumKronMats, M, N, K, KronMatCols, KronMatRows, 
                   &resultSize, &tempSize);
        T* sendTemp = (T*)temp1[g] + tempSize/2;
        T* recvTemp = (T*)temp2[g] + tempSize/2;
        //Call we want to use NCCL Send/Recv
        {
          const uint SliceRows = gpuM;
          const uint SliceCols = currTempN/handle.gpusInK_;
          const size_t sendRecvSize = SliceRows * SliceCols;
          const uint startRow = 0;
          const uint startCol = gc * SliceCols;
          matrixSlice(gpuM, currTempN, (T*)innerPrevResult, 
                      startRow, startCol, SliceRows, SliceCols,
                      recvTemp, stream[g], g, io, true);
          dim3 grid = {gpuM, 1,1};
          dim3 block = {256, 1, 1};
          storeGPUTile<T, 256><<<grid, block, 0, stream[g]>>>(M, currTempN*handle.gpusInK_, prevTempN*handle.gpusInK_,
                                                              KronMatRows[0], KronMatCols[0], gc, handle.gpusInK_,
                                                              (T*)recvTemp, gpuM, currTempN,
                                                              (T*)innerCurrResult, gc, KronMulBatchSize, io, distParams, false);
          CUDA_CHECK(cudaStreamSynchronize(stream[g]));
        }

        //All GPUs with the same gr share their intermediates
        for (int dst = 0; dst < handle.gpusInK_; dst++) {
          const uint SliceRows = gpuM;
          const uint SliceCols = currTempN/handle.gpusInK_;
          const size_t sendRecvSize = SliceRows * SliceCols;
          if (dst == gc) {
            for (int src = 0; src < handle.gpusInK_; src++) {
              // printf("g %d dst %d src %d\n", g, dst, src);
              if (src == dst) {
              } else {
                NCCLCHECK(ncclRecv(recvTemp, sendRecvSize, ncclFloat, gr * handle.gpusInK_ + src, handle.ncclComms[g], stream[g]));
                CUDA_CHECK(cudaStreamSynchronize(stream[g]));
                dim3 grid = {gpuM, 1,1};
                dim3 block = {256, 1, 1};
                storeGPUTile<T, 256><<<grid, block, 0, stream[g]>>>(M, currTempN*handle.gpusInK_, prevTempN*handle.gpusInK_,
                                                                    KronMatRows[0], KronMatCols[0], gc, handle.gpusInK_,
                                                                    (T*)recvTemp, gpuM, currTempN,
                                                                    (T*)innerCurrResult, src, KronMulBatchSize, io, distParams, false);
                CUDA_CHECK(cudaStreamSynchronize(stream[g]));
              }
            }
          } else {
            const uint startRow = 0;
            const uint startCol = dst * SliceCols;
            matrixSlice(gpuM, currTempN, (T*)innerPrevResult, 
                        startRow, startCol, SliceRows, SliceCols,
                        sendTemp, stream[g], g, io);
            CUDA_CHECK(cudaStreamSynchronize(stream[g]));
            // if (g == 1 && dst == 0) {
            //    printGPUArray<float>(SliceRows, SliceCols, (float*)handle.sendTemps_[g], stream[g]);
            //    printf("699 dst %d g %d\n", dst, g);
            // }
            NCCLCHECK(ncclSend(sendTemp, sendRecvSize, ncclFloat, gr * handle.gpusInK_ + dst, handle.ncclComms[g], stream[g]));
            CUDA_CHECK(cudaStreamSynchronize(stream[g]));
          }
        }

        innerPrevResult = innerCurrResult;
        if (innerPrevResult == innerResults[0]) {        
          innerCurrResult = innerResults[1];
        } else if (innerPrevResult == innerResults[1]) {
          innerCurrResult = innerResults[0];
        }
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream[g]));
    thread_barrier_wait(thArgs->barrier);
  }

  end:
  thArgs->threadResult = {status, (void*)innerPrevResult};
}

cudaError_t distributedKronMatmul(FastKronHandle& handle, const uint NumKronMats, void* x[], void* kronMats[], void* result[],
                                  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], void** temp1, void** temp2,
                                  cudaStream_t streams[]) {
  uint gpuM, gpuK;
  handle.getDistributedSizes(M, K, gpuM, gpuK);
  if (!checkDistributedKronSizes(NumKronMats, M, N, K, KronMatCols, KronMatRows, handle.perGPUKronBatch_, handle.gpusInK_))
    return cudaErrorInvalidValue;

  if (result == NULL)                        return cudaErrorInvalidValue;
  if (M % gpuM != 0)                         return cudaErrorInvalidValue;
  if (NumKronMats < handle.perGPUKronBatch_) return cudaErrorInvalidValue;
  if (temp1 == nullptr)                      return cudaErrorInvalidValue;

  const uint batchedKronMuls = handle.perGPUKronBatch_;

  thread_pool<ThreadArgs*>::task tasks[handle.numGPUs_];
  ThreadArgs threadArgs[handle.numGPUs_];

  for (uint thread = 0; thread < handle.numGPUs_; thread++) {
    ThreadArgs args = ThreadArgs(
      &handle,
      NumKronMats,
      (void*)x[thread],
      (void**)kronMats,
      (void**)result,
      M, N, K,
      &KronMatCols[0],
      &KronMatRows[0],
      (void**)temp1, (void**)temp2,
      streams,
      thread/handle.gpusInK_,
      thread % handle.gpusInK_,
      handle.gpusInM_,
      handle.gpusInK_,
      &handle.barriers_[thread/handle.gpusInK_]
    );

    threadArgs[thread] = args;
    tasks[thread] = thread_pool<ThreadArgs*>::task(perGPUKronMatmul, &threadArgs[thread]);
  }

  handle.threads_->execute_tasks(tasks);
  handle.threads_->join_tasks();

  cudaError_t status;
  for (uint thread = 0; thread < handle.numGPUs_; thread++) {
    status = threadArgs[thread].threadResult.status;
    // result[thread] =(T*)threadArgs[thread].threadResult.result;
  }

  return status;
}

uint getYColumns(uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  size_t tempN = K;
  size_t maxTempN = tempN;
  for (int i = 0; i < NumKronMats; i++) {
    tempN = (tempN/KronMatRows[i])*KronMatCols[i];
    if (maxTempN < tempN)
      maxTempN = tempN;
  }

  return tempN;
}

cudaError_t FastKronHandle::allocDistributedX(void* dX[], void* hX, uint M, uint K) {
  //TODO: Make FastKronError type
  if (!isDistributed_) return cudaErrorInvalidValue;
  uint gpuM, gpuK;
  getDistributedSizes(M, K, gpuM, gpuK);
  //TODO: Check that hX is on host memory
  typedef float T;
  T* gpuHostX = new T[((size_t)gpuM) * ((size_t)gpuK)];
  std::cout << "Distributing X to all GPUs "<<std::endl;
  // std::cout << handle.gpuM_ << "  " << handle.gpuK_ << "  " << sizeof(T) << std::endl;
  for (int g = 0; g < numGPUs_; g++) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMalloc(&dX[g], sizeof(T) * gpuM * gpuK));
  }

  for(int gr = 0; gr < gpusInM_; gr++) {
    for (uint gc = 0; gc < gpusInK_; gc++) {
      const uint g = gr * gpusInK_ + gc;
      // std::cout << "g " << g << " gr " <<gr << " gc " << gc << std::endl;
      CUDA_CHECK(cudaSetDevice(g));
      uint startGpuM = gpuM * gr;
      uint startGpuK = gpuK * gc;
        
      for (uint m = 0; m < gpuM; m++) {
        std::memcpy(&gpuHostX[m * gpuK], &((T*)hX)[(startGpuM+m)*K + startGpuK], sizeof(T)*gpuK);
      }
      CUDA_CHECK(cudaMemcpy(dX[g], gpuHostX, sizeof(T) * gpuM * gpuK, cudaMemcpyHostToDevice));
    }
  }
  delete gpuHostX;
  std::cout << "Distributed X " << std::endl;
  return cudaSuccess;
}

cudaError_t FastKronHandle::gatherDistributedY(void* dY[], void* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
  //TODO: Make FastKronError type
  typedef float T;
  if (!isDistributed_) return cudaErrorInvalidValue;
  //TODO: Check that hY is on host memory
  uint gpuM, gpuYCols, YCols;
  YCols = getYColumns(M, K, NumKronMats, KronMatCols, KronMatRows);
  getDistributedSizes(M, YCols, gpuM, gpuYCols);
  T* gpuHostY = new T[gpuM * gpuYCols];
  std::cout << "Gather Y from all GPUs"<<std::endl;

  for(int gr = 0; gr < gpusInM_; gr++) {
    for (uint gc = 0; gc < gpusInK_; gc++) {
      uint g = gr * gpusInK_ + gc;
      CUDA_CHECK(cudaSetDevice(g));
      //TODO: check that dX[g] is on GPU g
      CUDA_CHECK(cudaMemcpy(gpuHostY, dY[g], 
                            sizeof(T) * gpuM * gpuYCols,
                            cudaMemcpyDeviceToHost));
      const uint startGpuM = gpuM * gr;
      const uint startGpuN = gpuYCols * gc;
      for (int m = 0; m < gpuM; m++) {
        std::memcpy(&((T*)hY)[(startGpuM+m)*YCols + startGpuN],
                    &gpuHostY[m * gpuYCols], sizeof(T)*gpuYCols);
      }
    }
  }
  
  delete gpuHostY;

  std::cout << "Gathered Y" << std::endl;

  return cudaSuccess;
}

cudaError_t FastKronHandle::distributedsgekmm(const uint NumKronMats, float* x[], float* kronMats[], float* result[],
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
  cudaStream_t streams[]) {
    return distributedKronMatmul(*this, NumKronMats, (void*)x, (void**)kronMats, (void*)result, M, N, K, 
      KronMatCols, KronMatRows, (void**)temp1, (void**)temp2, streams);
}

FastKronHandle::FastKronHandle(int gpus, int gpusInM, int gpusInK, int gpuKrons) : tunedKernelSeries() {
  //TODO: Support both modes. Single Process multi gpu and multi process multi gpu
  useFusion_ = true;
  isDistributed_ = gpus > 1;
  numGPUs_ = gpus;

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
    // gpuM_ = M_/gpusInM_;
    // gpuK_ = K_/gpusInK_;
    // gpuN_ = N_/gpusInK_;
    
    //All gpus with same row shares the same barrier
    //TODO: free
    barriers_ = new pthread_barrier_t[gpusInM_];
    threads_ = new thread_pool<ThreadArgs*>(numGPUs_);

    for (int i = 0; i < gpusInM_; i++) {
      int s = pthread_barrier_init(&barriers_[i], NULL, gpusInK_);
      PTHREAD_BARRIER_CHECK(s);
    }
    
    // size_t tempN = gpuK_;
    // size_t maxTempN = tempN;
    // for (int i = 0; i < NumKronMats_; i++) {
    //   tempN = (tempN/KronMatRows_[i])*KronMatCols_[i];
    //   if (maxTempN < tempN)
    //     maxTempN = tempN;
    // }

    // size_t sz = gpuM_ * maxTempN * sizeof(T);
    // std::cout << "Allocating temporaries of size "<< sz << std::endl;
    // std::cout << "Allocated temporaries"<<std::endl;

  }

  //Load kernels into compiledKernels map
  for (uint i = 0; i < sizeof(KronGemmKernels)/sizeof(KernelInfo); i++) {
    KernelInfo& info = KronGemmKernels[i];
    KronMatmulShape shape {info.KronCols, info.KronRows, info.MaxColsA, 0, info.NumFusedKerns, info.DistributeToGPUs};
    auto iter = compiledKernels.find(shape);
    if (iter == compiledKernels.end()) {
      compiledKernels.emplace(std::make_pair(shape, std::vector<KernelInfo>()));
    }
    compiledKernels.at(shape).push_back(info);
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

void FastKronHandle::free() {
  if (isDistributed_) {
    for (uint g = 0; g < gpusInM_; g++) {
      int s = pthread_barrier_destroy(&barriers_[g]);
      PTHREAD_BARRIER_CHECK(s);
    }

    delete threads_;
    delete barriers_;

    if (distComm_ == DistComm::NCCL) {
      for (int i=0; i<ncclComms.size(); i++)
        ncclCommDestroy(ncclComms[i]);
    }
  }
  compiledKernels.clear();
}

void FastKronHandle::getDistributedSizes(uint M, uint K, uint& gpuM, uint& gpuK) {
  gpuM = M/gpusInM_;
  gpuK = K/gpusInK_;
}