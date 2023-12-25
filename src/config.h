#if defined(__NVCC__) || defined(__CUDACC__)
    #define CUDA_DEVICE_HOST __host__ __device__ __forceinline__
    #define CUDA_HOST __host__
    #define CUDA_DEVICE __device__ __forceinline__
#else
    #define CUDA_DEVICE_HOST
    #define CUDA_HOST
    #define CUDA_DEVICE
#endif