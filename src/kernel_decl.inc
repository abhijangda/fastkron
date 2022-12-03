static uint MaxTileRowsA[] = {1, 1, 1, 1, 2, 2};

#define KERNEL_DECL(T, VecT, K_EQUALS_VAR) \
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 16, 2, 2, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 16, 4, 4, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 16, 8, 8, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 16, 16, 16, 32, K_EQUALS_VAR, 1, 1, 1>,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 32, 2, 2, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 32, 4, 4, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 32, 8, 8, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 32, 16, 16, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 32, 32, 32, 32, K_EQUALS_VAR, 1, 1, 8>,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 64, 2, 2, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 64, 4, 4, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 64, 8, 8, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 64, 16, 16, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 64, 32, 32, 32, K_EQUALS_VAR, 1, 1, 8>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 64, 64, 64, 32, K_EQUALS_VAR, 1, 1, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 128, 2, 2, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 128, 4, 4, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 128, 8, 8, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 128, 16, 16, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 128, 32, 32, 32, K_EQUALS_VAR, 1, 1, 8>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 128, 64, 64, 32, K_EQUALS_VAR, 1, 1, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 256, 2, 2, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 256, 4, 4, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 256, 8, 8, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 256, 16, 16, 32, K_EQUALS_VAR, 1, 1, 1>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 256, 32, 32, 32, K_EQUALS_VAR, 1, 1, 8>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 256, 64, 64, 32, K_EQUALS_VAR, 1, 1, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 512, 2, 2, 32, K_EQUALS_VAR, 1, 1, 2>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 512, 4, 4, 32, K_EQUALS_VAR, 1, 1, 2>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 512, 8, 8, 32, K_EQUALS_VAR, 1, 1, 2>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 512, 16, 16, 32, K_EQUALS_VAR, 1, 1, 2>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 512, 32, 32, 32, K_EQUALS_VAR, 1, 1, 8>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 512, 64, 64, 32, K_EQUALS_VAR, 1, 1, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 1024, 2, 2, 32, K_EQUALS_VAR, 1, 2, 2>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 1024, 4, 4, 32, K_EQUALS_VAR, 1, 1, 4>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 1024, 8, 8, 32, K_EQUALS_VAR, 1, 1, 4>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 1024, 16, 16, 32, K_EQUALS_VAR, 1, 1, 4>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 1024, 32, 32, 32, K_EQUALS_VAR, 1, 1, 8>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 1024, 64, 64, 32, K_EQUALS_VAR, 1, 1, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 2048, 2, 2, 32, K_EQUALS_VAR, 1, 4, 2>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 2048, 4, 4, 32, K_EQUALS_VAR, 1, 2, 4>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 2048, 8, 8, 32, K_EQUALS_VAR, 1, 1, 8>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 2048, 16, 16, 32, K_EQUALS_VAR, 1, 1, 8>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 2048, 32, 32, 32, K_EQUALS_VAR, 1, 1, 8>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 2048, 64, 64, 32, K_EQUALS_VAR, 1, 1, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 4096, 2, 2, 32, K_EQUALS_VAR, 1, 4, 2>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 4096, 4, 4, 32, K_EQUALS_VAR, 1, 2, 4>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 4096, 8, 8, 32, K_EQUALS_VAR, 1, 1, 8>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 4096, 16, 16, 32, K_EQUALS_VAR, 1, 1, 16>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 4096, 32, 32, 32, K_EQUALS_VAR, 1, 1, 8>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 4096, 64, 64, 32, K_EQUALS_VAR, 1, 1, 32>
