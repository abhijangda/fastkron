#include "kernel_invoker.h"
#include "utils.h"

static bool isValidKernel(KernelInfo& kernelInfo) {
  const uint NumThreads = kernelInfo.NumThreads;
  const uint KronRows = kernelInfo.KronRows;
  const uint KronCols = kernelInfo.KronCols;
  const uint CRegRows = kernelInfo.CRegRows;
  const uint CRegCols = kernelInfo.CRegCols;
  const uint MaxColsA = kernelInfo.MaxColsA;
  const uint TileKronCols = kernelInfo.TileKronCols;

  const uint ValidThreads = ((MaxColsA/KronRows)/CRegRows) * (TileKronCols/CRegCols);
  if (NumThreads != ROUNDUP(ValidThreads, CUDA_WARP_SIZE)) {
    std::cout << "Invalid kernel config " << kernelInfo << std::endl; 
    return false;
  }

  return true;
}

//Launch cuda kernels
template<uint NumFusedKerns>
cudaError_t generalSlicedMatmul(KernelInfo& kernelInfo, const uint kronIndex, 
                                void* x, void** kronMat, void* kronGemmResult,
                                const uint M, const uint N, const uint K, 
                                const uint KronMatCols[NumFusedKerns], const uint KronMatRows[NumFusedKerns],
                                EpilogueParams epilogueParams,
                                cudaStream_t stream) {
  cudaError_t status;
  
  if (!isValidKernel(kernelInfo)) abort();
  
  //Create the grid and thread block
  dim3 grid;
  dim3 block;
  grid = {
          (K/kernelInfo.MaxColsA) * DIVUP(KronMatCols[0], kernelInfo.TileKronCols),
          DIVUP(M, kernelInfo.TileRowsA),
          1
         };
  block = {
            kernelInfo.NumThreads, 
            1, 
            1
          };
  
  KernelParams<NumFusedKerns> params (M, N, K,
                                      KronMatRows, 
                                      KronMatCols,
                                      x, 
                                      kronMat, 
                                      kronGemmResult, 
                                      kronIndex);
  FusedParams<NumFusedKerns> fusedParams (M, N, K, kernelInfo.MaxColsA, KronMatRows, KronMatCols);
  // std::cout << "Invoking " << kernelInfo << std::endl;
  //Call kernel
  typedef void (*KronMatmulKernelTy)(KernelParams<NumFusedKerns>, FusedParams<NumFusedKerns>, 
                                     DistributedParams, EpilogueParams, dim3, dim3, cudaStream_t);
  KronMatmulKernelTy(kernelInfo.kernel)(params, fusedParams, DistributedParams(), 
                                        epilogueParams, grid, block, stream);
  status = cudaGetLastError();
  CUDA_CHECK(status);
  return status;
}

cudaError_t KernelInvoker::fusedSlicedMatmul(uint NumFusedKerns, KernelInfo& kernelInfo, const uint kronIndex, 
                              void* x, void** krons, void* kronGemmResult,
                              const uint M, const uint N, const uint K, 
                              const uint* FusedKronMatCols, const uint* FusedKronMatRows,
                              EpilogueParams epilogueParams,
                              cudaStream_t stream) {
  switch(NumFusedKerns) {
    case 1:
      return generalSlicedMatmul<1>(kernelInfo, kronIndex, x,
                                        krons, kronGemmResult, M, N, K,
                                        FusedKronMatCols, FusedKronMatRows,
                                        epilogueParams, stream);
    case 2:
      return generalSlicedMatmul<2>(kernelInfo, kronIndex, x,
                                          krons, kronGemmResult, M, N, K,
                                          FusedKronMatCols, FusedKronMatRows,
                                          epilogueParams, stream);
    case 3:
      return generalSlicedMatmul<3>(kernelInfo, kronIndex, x,
                                          krons, kronGemmResult, M, N, K,
                                          FusedKronMatCols, FusedKronMatRows,
                                          epilogueParams, stream);
    case 4:
      return generalSlicedMatmul<4>(kernelInfo, kronIndex, x,
                                          krons, kronGemmResult, M, N, K,
                                          FusedKronMatCols, FusedKronMatRows,
                                          epilogueParams, stream);
    case 5:
      return generalSlicedMatmul<5>(kernelInfo, kronIndex, x,
                                          krons, kronGemmResult, M, N, K,
                                          FusedKronMatCols, FusedKronMatRows,
                                          epilogueParams, stream);
      break;
    default:
        std::cout << "Invalid number of fused kernels" << std::endl;
      return cudaErrorInvalidValue;
  }
}

//Launch cuda kernels
template<uint NumFusedKerns>
static cudaError_t generalDistributedSlicedMatmul(KernelInfo& kernelInfo, const uint kronIndex, 
                                           void* x, void** kronMat, void* kronGemmResult,
                                           const uint M, const uint N, const uint K, 
                                           const uint KronMatCols[NumFusedKerns], const uint KronMatRows[NumFusedKerns],
                                           DistributedParams distParams, EpilogueParams epilogueParams,
                                           cudaStream_t stream) {
  cudaError_t status;
  
  if (!isValidKernel(kernelInfo)) abort();

  //Create the grid and thread block
  dim3 grid;
  dim3 block;
  
  grid = {
          (K/kernelInfo.MaxColsA) * DIVUP(KronMatCols[0], kernelInfo.TileKronCols),
          DIVUP(M, kernelInfo.TileRowsA),
          1
         };
  block = {
            kernelInfo.NumThreads, 
            1, 
            1
          };

  KernelParams<NumFusedKerns> params(M, N, K,
                                     KronMatRows, 
                                     KronMatCols, 
                                     (void*)x, 
                                     (void**)kronMat, 
                                     (void*)kronGemmResult, 
                                     kronIndex);
  FusedParams<NumFusedKerns> fusedParams(M, N, K, kernelInfo.MaxColsA, KronMatRows, KronMatCols);

  //Call kernel
  //TODO: No need to have Type template (T) as part of Kernelparams and DistributedParams
  typedef void (*KronMatmulKernelTy)(KernelParams<NumFusedKerns>, FusedParams<NumFusedKerns>, 
                                     DistributedParams, EpilogueParams, dim3, dim3, cudaStream_t);
  KronMatmulKernelTy(kernelInfo.kernel)(params, fusedParams, distParams, epilogueParams, 
                                        grid, block, stream);
  status = cudaGetLastError();
  CUDA_CHECK(status);
  return status;
}

cudaError_t KernelInvoker::fusedDistributedSlicedMatmul(const uint NumFusedKerns, KernelInfo& kernel, const uint kronIndex, 
                                           void* x, void** kronMat, void* kronGemmResult,
                                           const uint M, const uint N, const uint K, 
                                           const uint* FusedKronMatCols, const uint* FusedKronMatRows,
                                           DistributedParams distParams, EpilogueParams epilogueParams,
                                           cudaStream_t stream) {
  switch (NumFusedKerns) {
    case 1:
      return generalDistributedSlicedMatmul<1>(kernel, kronIndex, x, 
                                                  kronMat, kronGemmResult, M, N, K, 
                                                  FusedKronMatCols, FusedKronMatRows, 
                                                  distParams, epilogueParams, stream);
    case 2:
      return generalDistributedSlicedMatmul<2>(kernel, kronIndex, x, 
                                                    kronMat, kronGemmResult, M, N, K,
                                                    FusedKronMatCols, FusedKronMatRows, 
                                                    distParams, epilogueParams, stream);
    case 3:
      return generalDistributedSlicedMatmul<3>(kernel, kronIndex, x, 
                                                    kronMat, kronGemmResult, M, N, K,
                                                    FusedKronMatCols, FusedKronMatRows, 
                                                    distParams, epilogueParams, stream);
    case 4:
      return generalDistributedSlicedMatmul<4>(kernel, kronIndex, x, 
                                                    kronMat, kronGemmResult, M, N, K,
                                                    FusedKronMatCols, FusedKronMatRows, 
                                                    distParams, epilogueParams, stream);
    case 5:
      return generalDistributedSlicedMatmul<5>(kernel, kronIndex, x, 
                                                    kronMat, kronGemmResult, M, N, K, 
                                                    FusedKronMatCols, FusedKronMatRows, 
                                                    distParams, epilogueParams, stream);
  }

  return cudaErrorInvalidValue;
}