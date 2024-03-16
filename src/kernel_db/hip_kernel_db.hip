#include "hip_kernel_db.h"

#include <hip/hip_runtime_api.h>


//Launch cuda kernels
template<uint FusedFacs>
fastKronError invoke(CUDAKernel& kernelInfo, const uint kronIndex, 
                   KMMProblem problem,
                   DistributedParams distParams,
                   EpilogueParams epilogueParams,
                   KernelMode execMode,
                   hipStream_t stream) {
  hipError_t status;

  //Create the grid and thread block
  KernelParams<FusedFacs> params (problem, kronIndex, execMode);
  FusedParams<FusedFacs> fusedParams (problem, kernelInfo.tileX.n());

  //Call kernel
  typedef void (*KronMatmulKernelTy)(KernelParams<FusedFacs>, FusedParams<FusedFacs>, 
                                     DistributedParams, EpilogueParams, dim3, dim3, uint32_t, hipStream_t);
  KronMatmulKernelTy(kernelInfo.invokerFunc)(params, fusedParams, distParams, 
                                        epilogueParams, kernelInfo.grid(problem), 
                                        kernelInfo.block(), kernelInfo.sharedMemSize(), stream);
  status = hipGetLastError();
  HIP_CHECK(status);
  return fastKronSuccess;
}

fastKronError HIPKernelDatabase::invokeKernel(KernelInfo* kernel, const uint kronIndex, 
                                             KMMProblem problem, EpilogueParams epilogueParams,
                                             KernelMode execMode) {
  DistributedParams distParams;
  hipStream_t stream = (hipStream_t*)streams[0];
  HIPKernel& hipKernel = dynamic_cast<HIPKernel&>(*kernel);

  switch(problem.n()) {
    case 1:
      return invoke<1>(hipKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 2:
      return invoke<2>(hipKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 3:
      return invoke<3>(hipKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 4:
      return invoke<4>(hipKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 5:
      return invoke<5>(hipKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode, stream);
    case 6:
      return invoke<6>(hipKernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode, stream);
    default:
      std::cout << "Invalid number of fused kernels" << std::endl;
      return fastKronErrorKernelNotFound;
  }
}

fastKronError HIPKernelDatabase::procMalloc(uint32_t proc, size_t size, void*& ptr){
  HIP_CHECK(hipSetDevice(proc));
  HIP_CHECK(hipMalloc(&ptr, size));
  HIP_CHECK(hipMemset(ptr, 1, size));
  
  return fastKronSuccess;
}

fastKronError HIPKernelDatabase::procMemset(uint32_t proc, Matrix& m, float val) {
  HIP_CHECK(hipSetDevice(proc));
  float* host = new float[m.numel()];
  memset<float>(host, m.numel(), val);
  HIP_CHECK(hipMemcpy(m.data(), host, m.numel()*sizeof(float), hipMemcpyHostToDevice));
  delete[] host;

  return fastKronSuccess;
}

fastKronError HIPKernelDatabase::procFree(uint32_t proc, void* ptr) {
  HIP_CHECK(hipSetDevice(proc));
  HIP_CHECK(hipFree(ptr));

  return fastKronSuccess;
}