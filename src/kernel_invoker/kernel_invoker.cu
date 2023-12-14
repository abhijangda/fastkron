#include "kernel_invoker/kernel_invoker.h"
#include "utils/utils.h"


//Launch cuda kernels
template<uint NumFusedKerns>
cudaError_t invoke(KernelInfo& kernelInfo, const uint kronIndex, 
                   KMMProblem problem,
                   DistributedParams distParams,
                   EpilogueParams epilogueParams,
                   cudaStream_t stream) {
  cudaError_t status;

  //Create the grid and thread block
  KernelParams<NumFusedKerns> params (problem, kronIndex);
  FusedParams<NumFusedKerns> fusedParams (problem, kernelInfo.tiledInput.N);

  //Call kernel
  typedef void (*KronMatmulKernelTy)(KernelParams<NumFusedKerns>, FusedParams<NumFusedKerns>, 
                                     DistributedParams, EpilogueParams, dim3, dim3, cudaStream_t);
  KronMatmulKernelTy(kernelInfo.kernel)(params, fusedParams, distParams, 
                                        epilogueParams, kernelInfo.grid(problem), 
                                        kernelInfo.block(), stream);
  status = cudaGetLastError();
  CUDA_CHECK(status);
  return status;
}

cudaError_t KernelInvoker::invokeKernel(KernelInfo& kernelInfo, const uint kronIndex, 
                                             KMMProblem problem, EpilogueParams epilogueParams,
                                             cudaStream_t stream) {
  DistributedParams distParams;

  switch(problem.n) {
    case 1:
      return invoke<1>(kernelInfo, kronIndex, problem,
                       distParams, epilogueParams, stream);
    case 2:
      return invoke<2>(kernelInfo, kronIndex, problem,
                       distParams, epilogueParams, stream);
    case 3:
      return invoke<3>(kernelInfo, kronIndex, problem,
                       distParams, epilogueParams, stream);
    case 4:
      return invoke<4>(kernelInfo, kronIndex, problem,
                       distParams, epilogueParams, stream);
    case 5:
      return invoke<5>(kernelInfo, kronIndex, problem,
                       distParams, epilogueParams, stream);
      break;
    default:
        std::cout << "Invalid number of fused kernels" << std::endl;
      return cudaErrorInvalidValue;
  }
}

cudaError_t KernelInvoker::invokeP2PStoreKernel(KernelInfo& kernel, const uint kronIndex, 
                                                KMMProblem problem, DistributedParams distParams, 
                                                EpilogueParams epilogueParams,
                                                cudaStream_t stream) {
  switch (problem.n) {
    case 1:
      return invoke<1>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, stream);
    case 2:
      return invoke<2>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, stream);
    case 3:
      return invoke<3>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, stream);
    case 4:
      return invoke<4>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, stream);
    case 5:
      return invoke<5>(kernel, kronIndex, problem, 
                       distParams, epilogueParams, stream);
  }

  return cudaErrorInvalidValue;
}