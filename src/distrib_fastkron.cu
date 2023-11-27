#include "handle.h"

static void thread_barrier_wait(pthread_barrier_t* barrier) {
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

static uint getYColumns(uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]) {
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
