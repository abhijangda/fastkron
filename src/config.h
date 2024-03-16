#if defined(__NVCC__) || defined(__CUDACC__) || defined(__HIPCC__)
    #if defined(__HIPCC__) && !defined(__forceinline__)
        #define __forceinline__ inline
    #endif

    #define CUDA_DEVICE_HOST __host__ __device__ __forceinline__
    #define CUDA_HOST __host__
    #define CUDA_DEVICE __device__ __forceinline__
#else
    #define CUDA_DEVICE_HOST
    #define CUDA_HOST
    #define CUDA_DEVICE
#endif