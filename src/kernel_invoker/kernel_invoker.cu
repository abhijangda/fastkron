#include "kernel_invoker/kernel_invoker.h"
#include "utils/utils.h"

static bool isValidKernel(KernelInfo& kernelInfo) {
  const uint NumThreads = kernelInfo.NumThreads;
  const SlicedMulShape tiledShape = kernelInfo.tiledShape;
  const uint CRegRows = kernelInfo.CRegRows;
  const uint CRegCols = kernelInfo.CRegCols;

  const uint ValidThreads = ((tiledShape.K/tiledShape.P)/CRegRows) * (tiledShape.Q/CRegCols);
  if (NumThreads != ROUNDUP(ValidThreads, CUDA_WARP_SIZE)) {
    std::cout << "Invalid kernel config " << kernelInfo << std::endl; 
    return false;
  }

  return true;
}

//Launch cuda kernels
template<uint NumFusedKerns>
cudaError_t generalSlicedMatmul(KernelInfo& kernelInfo, const uint kronIndex, 
                                KMMProblem problem,
                                EpilogueParams epilogueParams,
                                cudaStream_t stream) {
  cudaError_t status;
  
  if (!isValidKernel(kernelInfo)) abort();

  const SlicedMulShape tiledShape = kernelInfo.tiledShape;

  //Create the grid and thread block
  dim3 grid;
  dim3 block;
  grid = {
          (problem.k/tiledShape.K) * DIVUP(problem.qs[0], tiledShape.Q),
          DIVUP(problem.m, tiledShape.M),
          1
         };
  block = {
            kernelInfo.NumThreads, 
            1, 
            1
          };

  KernelParams<NumFusedKerns> params (problem, kronIndex);
  FusedParams<NumFusedKerns> fusedParams (problem, tiledShape.K);
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

cudaError_t KernelInvoker::fusedSlicedMatmul(KernelInfo& kernelInfo, const uint kronIndex, 
                                             KMMProblem problem, EpilogueParams epilogueParams,
                                             cudaStream_t stream) {
  switch(problem.n) {
    case 1:
      return generalSlicedMatmul<1>(kernelInfo, kronIndex, problem,
                                    epilogueParams, stream);
    case 2:
      return generalSlicedMatmul<2>(kernelInfo, kronIndex, problem,
                                    epilogueParams, stream);
    case 3:
      return generalSlicedMatmul<3>(kernelInfo, kronIndex, problem,
                                    epilogueParams, stream);
    case 4:
      return generalSlicedMatmul<4>(kernelInfo, kronIndex, problem,
                                    epilogueParams, stream);
    case 5:
      return generalSlicedMatmul<5>(kernelInfo, kronIndex, problem,
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
                                                  KMMProblem problem,
                                                  DistributedParams distParams, EpilogueParams epilogueParams,
                                                  cudaStream_t stream) {
  cudaError_t status;
  const SlicedMulShape tiledShape = kernelInfo.tiledShape;

  if (!isValidKernel(kernelInfo)) abort();

  //Create the grid and thread block
  dim3 grid;
  dim3 block;
  grid = {
          (problem.k/tiledShape.K) * DIVUP(problem.qs[0], tiledShape.Q),
          DIVUP(problem.m, tiledShape.M),
          1
         };
  block = {
            kernelInfo.NumThreads, 
            1, 
            1
          };

  KernelParams<NumFusedKerns> params (problem, kronIndex);
  FusedParams<NumFusedKerns> fusedParams (problem, tiledShape.K);

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

cudaError_t KernelInvoker::fusedDistributedSlicedMatmul(KernelInfo& kernel, const uint kronIndex, 
                                                        KMMProblem problem, DistributedParams distParams, 
                                                        EpilogueParams epilogueParams,
                                                        cudaStream_t stream) {
  switch (problem.n) {
    case 1:
      return generalDistributedSlicedMatmul<1>(kernel, kronIndex, problem, 
                                               distParams, epilogueParams, stream);
    case 2:
      return generalDistributedSlicedMatmul<2>(kernel, kronIndex, problem, 
                                               distParams, epilogueParams, stream);
    case 3:
      return generalDistributedSlicedMatmul<3>(kernel, kronIndex, problem, 
                                               distParams, epilogueParams, stream);
    case 4:
      return generalDistributedSlicedMatmul<4>(kernel, kronIndex, problem, 
                                               distParams, epilogueParams, stream);
    case 5:
      return generalDistributedSlicedMatmul<5>(kernel, kronIndex, problem, 
                                               distParams, epilogueParams, stream);
  }

  return cudaErrorInvalidValue;
}