#if defined(__NVCC__) || defined(__CUDACC__) || defined(__HIPCC__)
    #if !defined(__forceinline__)
        #define __forceinline__ inline
    #endif

    #define CUDA_HOST __host__
    #define CUDA_DEVICE __device__ __forceinline__
    #define CUDA_DEVICE_HOST CUDA_HOST CUDA_DEVICE
#else
    #define CUDA_HOST
    #define CUDA_DEVICE inline
    #define CUDA_DEVICE_HOST CUDA_HOST CUDA_DEVICE
#endif

#define PRAGMA(X) _Pragma(#X)

#if defined(__clang__)
    #define CXX_PRAGMA_PUSH_OPTIONS   _Pragma("")
    #define CXX_PRAGMA_O3
    #define CXX_PRAGMA_ARCH_SISD      _Pragma("clang attribute push (__attribute__((target(\"arch=x86-64-v2\"))), apply_to=function)")
    #define CXX_PRAGMA_ARCH_AVX       _Pragma("clang attribute push (__attribute__((target(\"arch=x86-64-v3\"))), apply_to=function)")
    #define CXX_PRAGMA_ARCH_AVX512    _Pragma("clang attribute push (__attribute__((target(\"arch=x86-64-v4\"))), apply_to=function)")
    #define CXX_PRAGMA_POP_OPTIONS    _Pragma("clang attribute pop")

#elif defined(__GNUC__) || defined(__GNUG__)
    #define CXX_PRAGMA_PUSH_OPTIONS   _Pragma("GCC push_options")
    #define CXX_PRAGMA_O3             _Pragma("GCC optimization(\"O3\")")
    #define CXX_PRAGMA_ARCH_SISD      _Pragma("GCC target(\"arch=x86-64-v2\")")
    #define CXX_PRAGMA_ARCH_AVX       _Pragma("GCC target(\"arch=x86-64-v3\")")
    #define CXX_PRAGMA_ARCH_AVX512    _Pragma("GCC target(\"arch=x86-64-v4\")")
    #define CXX_PRAGMA_POP_OPTIONS    _Pragma("GCC pop_options")
#endif