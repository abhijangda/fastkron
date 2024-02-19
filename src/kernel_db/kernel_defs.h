#ifdef ENABLE_CUDA
  #include "kernels/cuda/kron-kernels/kernel_decl.inc"
#endif

CUDAKernel AllCUDAKernels[] = {
#ifdef ENABLE_CUDA
  ALL_CUDA_KERNELS
#endif
};
