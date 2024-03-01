#include "kernels/params.h"
#include "kernel_db/cpu_kernel_db.h"

#ifdef ENABLE_X86
  #include "kernels/cpu/x86/kron-kernels/kernel_decl.inc"
#endif

CPUKernel AllX86Kernels[] = {
#ifdef ENABLE_X86
  ALL_X86_KERNELS
#endif
};

CPUKernelDatabase::CPUKernelDatabase() : KernelDatabase() {
  loadKernels<CPUKernel>(AllX86Kernels, sizeof(AllX86Kernels)/sizeof(CPUKernel));
}

template<uint NumFusedKerns>
cudaError_t invoke(CPUKernel& kernelInfo, const uint kronIndex, 
                   KMMProblem problem,
                   DistributedParams distParams,
                   EpilogueParams epilogueParams,
                   KernelMode execMode) {
  cudaError_t status;

  //Create the grid and thread block
  KernelParams<NumFusedKerns> params (problem, kronIndex, execMode);
  FusedParams<NumFusedKerns> fusedParams (problem, kernelInfo.tiledInput.n());

  //Call kernel
  typedef void (*KronMatmulKernelTy)(KernelParams<NumFusedKerns>, FusedParams<NumFusedKerns>, 
                                     DistributedParams, EpilogueParams);
  KronMatmulKernelTy(kernelInfo.invokerFunc)(params, fusedParams, distParams, epilogueParams);
  status = cudaSuccess;
  CUDA_CHECK(status);
  return status;
}

cudaError_t CPUKernelDatabase::invokeKernel(KernelInfo* kernel, const uint kronIndex, 
                                            KMMProblem problem,
                                            EpilogueParams epilogueParams,
                                            KernelMode execMode) {
  DistributedParams distParams;
  CPUKernel& cpuKernel = dynamic_cast<CPUKernel&>(*kernel);

  switch(problem.n()) {
    case 1:
      return invoke<1>(cpuKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode);
    case 2:
      return invoke<2>(cpuKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode);
    case 3:
      return invoke<3>(cpuKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode);
    case 4:
      return invoke<4>(cpuKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode);
    case 5:
      return invoke<5>(cpuKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode);
    case 6:
      return invoke<6>(cpuKernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode);
    default:
      std::cout << "Invalid number of fused kernels" << std::endl;
      return cudaErrorInvalidValue;
  }
}

cudaError_t CPUKernelDatabase::procMalloc(uint32_t proc, size_t size, void*& ptr) {
  ptr = new float[size];
  return ptr != nullptr ? cudaSuccess : cudaErrorInvalidValue; 
}

cudaError_t CPUKernelDatabase::procMemset(uint32_t proc, Matrix& m, float val) {
  memset<float>(m.data<float>(0), m.numel(), val);
  return cudaSuccess;
}

cudaError_t CPUKernelDatabase::procFree(uint32_t proc, void* ptr) {
  delete ptr;
  return cudaSuccess;
}

cudaError_t CPUKernelDatabase::timeKernel(KernelInfo* kernel, const uint factorIdx, 
                                 KMMProblem problem, DistributedParams distParams, 
                                 EpilogueParams epilogueParams,
                                 KernelMode execMode, 
                                 bool distP2PStore,
                                 int warmups, int runs,
                                 float& runtime) {
  runtime = std::numeric_limits<float>::max();
  cudaError_t status;
  for (int sample = 0; sample < 10; sample++) {
    double startTime = getCurrTime();
    for (int r = 0; r < runs; r++) {
      if (distP2PStore) {
        status = invokeP2PStoreKernel(kernel, factorIdx, problem,
                                      distParams, epilogueParams, execMode);
      } else {
        status = invokeKernel(kernel, factorIdx, problem,
                              epilogueParams, execMode);
      }
    }

    CUDA_CHECK(status);
    double endTime = getCurrTime();

    if (status != cudaSuccess) {
      std::cout << "Error: " << cudaGetErrorString(status) << std::endl;
      return status;
    }

    runtime = std::min((float)((endTime - startTime)/1e3)/runs, runtime);
  }
  
  return status;
}