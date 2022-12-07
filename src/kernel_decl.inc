#define MAX_K 65536
#define MIN_K 16
#define MIN_KP_K 2
#define MAX_KP_K 1024
static uint MaxTileRowsA[] = {1, 1, 1, 1, 2, 2, 2, 1, 1, 1};

static uint MaxTileKronCols[] = {32, 32, 32, 32, 32, 64, 64, 64, 128, 256};

#define KERNEL_DECL(T, VecT, K_EQUALS_VAR) \
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 16, 2, 2, 32, K_EQUALS_VAR, 1, 1, 1, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 16, 4, 4, 32, K_EQUALS_VAR, 1, 1, 1, 32>,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 16, 16, 16, 32, K_EQUALS_VAR, 1, 1, 1, 32>,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 32, 2, 2, 32, K_EQUALS_VAR, 1, 1, 1, 32>,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 32, 32, 32, 32, K_EQUALS_VAR, 1, 1, 4, 32>,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 64, 2, 2, 32, K_EQUALS_VAR, 1, 1, 1, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 64, 4, 4, 32, K_EQUALS_VAR, 1, 1, 1, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 64, 8, 8, 32, K_EQUALS_VAR, 1, 1, 1, 32>,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 64, 64, 64, 64, K_EQUALS_VAR, 1, 1, 8, 16>,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 128, 2, 2, 32, K_EQUALS_VAR, 1, 1, 1, 32>,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 128, 128, 128, 64, K_EQUALS_VAR, 1, 1, 32, 16>,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 256, 2, 2, 32, K_EQUALS_VAR, 1, 1, 1, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 256, 4, 4, 32, K_EQUALS_VAR, 1, 1, 1, 32>,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 256, 16, 16, 32, K_EQUALS_VAR, 1, 1, 1, 32>,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 512, 2, 2, 32, K_EQUALS_VAR, 1, 1, 2, 32>,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 512, 8, 8, 32, K_EQUALS_VAR, 1, 1, 2, 32>,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 1024, 2, 2, 32, K_EQUALS_VAR, 1, 2, 2, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 1024, 4, 4, 32, K_EQUALS_VAR, 1, 1, 4, 32>,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 1024, 32, 32, 32, K_EQUALS_VAR, 1, 1, 4, 32>,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 1024, 1024, 1024, 256, K_EQUALS_VAR, 1, 1, 32, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 2048, 2, 2, 32, K_EQUALS_VAR, 1, 4, 2, 32>,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 4096, 2, 2, 32, K_EQUALS_VAR, 1, 4, 2, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 4096, 4, 4, 32, K_EQUALS_VAR, 1, 2, 4, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 4096, 8, 8, 32, K_EQUALS_VAR, 1, 1, 8, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 4096, 16, 16, 32, K_EQUALS_VAR, 1, 1, 16, 32>,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 4096, 64, 64, 64, K_EQUALS_VAR, 1, 1, 8, 16>,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 2, 16384, 128, 128, 64, K_EQUALS_VAR, 1, 1, 32, 16>,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 32768, 256, 256, 64, K_EQUALS_VAR, 1, 1, 32, 32>,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 32768, 512, 512, 128, K_EQUALS_VAR, 1, 1, 32, 32>,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    NULL,\
    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, 1, 65536, 1024, 1024, 256, K_EQUALS_VAR, 1, 1, 64, 32>
