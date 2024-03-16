#include "hip_kernel_db.h"

#include <hip/hip_runtime_api.h>

fastKronError HIPKernelDatabase::procMalloc(uint32_t proc, size_t size, void*& ptr){
    HIP_CHECK(hipSetDevice(proc));
    HIP_CHECK(hipMalloc(&ptr, size));
    HIP_CHECK(hipMemset(ptr, 1, size));
    
    return fastKronSuccess;
  }