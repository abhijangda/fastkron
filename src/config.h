#if defined(__NVCC__) || defined(__CUDACC__) || defined(__HIPCC__)
    #if !defined(__forceinline__)
        #define __forceinline__ inline
    #endif

    #define CUDA_HOST __host__
    #define CUDA_DEVICE __device__ __forceinline__
    #define CUDA_DEVICE_HOST CUDA_HOST CUDA_DEVICE
#else
    #define CUDA_HOST
    #define CUDA_DEVICE
    #define CUDA_DEVICE_HOST CUDA_HOST CUDA_DEVICE
#endif