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

cudaError_t CPUKernelDatabase::procMalloc(uint32_t proc, size_t size, void*& ptr) {
  ptr = new float[size];
  return ptr != nullptr ? cudaSuccess : cudaErrorInvalidValue; 
}

cudaError_t CPUKernelDatabase::procMemset(uint32_t proc, Matrix& m, float val) {
  memset<float>(m.data<float>(0), m.numel(), val);
}

cudaError_t CPUKernelDatabase::procFree(uint32_t proc, void* ptr) {

}