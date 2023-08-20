#define MAX_K 65536
#define MIN_K 16
#define MIN_KP_K 2
#define MAX_KP_K 1024
static uint MaxTileRowsA[] = {1, 1, 1, 1, 2, 1, 1, 1, 1, 1};

static uint MaxTileKronCols[] = {2, 4, 8, 16, 32, 64, 128, 64, 128, 256};

#define KERNEL_DECL(T, VecT, RowModTileIsZero, K_EQUALS_VAR) \
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 16, 2, 2, 2, K_EQUALS_VAR, 1, 1, 1, 32, 1>,256, 2, 2, 2, 16, 1, 1, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 16, 4, 4, 4, K_EQUALS_VAR, 1, 1, 1, 32, 1>,256, 4, 4, 4, 16, 1, 1, 1},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 16, 16, 16, 16, K_EQUALS_VAR, 1, 1, 1, 32, 1>,256, 16, 16, 16, 16, 1, 1, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 32, 2, 2, 2, K_EQUALS_VAR, 1, 1, 1, 32, 1>,256, 2, 2, 2, 32, 1, 1, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 2, RowModTileIsZero, 32, 32, 32, 32, K_EQUALS_VAR, 1, 1, 4, 32, 1>,256, 32, 32, 32, 32, 1, 4, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 64, RowParallelismTy::Low, 1, RowModTileIsZero, 64, 2, 2, 2, K_EQUALS_VAR, 1, 1, 1, 32, 1>,64, 2, 2, 2, 64, 1, 1, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 64, RowParallelismTy::Low, 1, RowModTileIsZero, 64, 4, 4, 4, K_EQUALS_VAR, 1, 1, 1, 32, 1>,64, 4, 4, 4, 64, 1, 1, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 64, RowParallelismTy::Low, 1, RowModTileIsZero, 64, 8, 8, 8, K_EQUALS_VAR, 1, 1, 1, 32, 1>,64, 8, 8, 8, 64, 1, 1, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 64, 64, 64, 64, K_EQUALS_VAR, 1, 1, 8, 32, 1>,256, 64, 64, 64, 64, 1, 8, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 128, RowParallelismTy::Low, 1, RowModTileIsZero, 128, 2, 2, 2, K_EQUALS_VAR, 1, 1, 1, 32, 1>,128, 2, 2, 2, 128, 1, 1, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 256, 2, 2, 2, K_EQUALS_VAR, 1, 1, 1, 32, 1>,256, 2, 2, 2, 256, 1, 1, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 256, 4, 4, 4, K_EQUALS_VAR, 1, 1, 1, 32, 1>,256, 4, 4, 4, 256, 1, 1, 1},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 256, 16, 16, 16, K_EQUALS_VAR, 1, 1, 1, 32, 1>,256, 16, 16, 16, 256, 1, 1, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 512, 2, 2, 2, K_EQUALS_VAR, 1, 1, 2, 32, 1>,256, 2, 2, 2, 512, 1, 2, 1},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 512, RowParallelismTy::Low, 1, RowModTileIsZero, 512, 8, 8, 8, K_EQUALS_VAR, 1, 1, 1, 32, 1>,512, 8, 8, 8, 512, 1, 1, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 512, RowParallelismTy::Low, 1, RowModTileIsZero, 1024, 2, 2, 2, K_EQUALS_VAR, 1, 1, 2, 32, 1>,512, 2, 2, 2, 1024, 1, 2, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 1024, 4, 4, 4, K_EQUALS_VAR, 1, 1, 4, 32, 1>,256, 4, 4, 4, 1024, 1, 4, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 2, RowModTileIsZero, 1024, 32, 32, 32, K_EQUALS_VAR, 1, 1, 4, 32, 1>,256, 32, 32, 32, 1024, 1, 4, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 1024, 1024, 1024, 256, K_EQUALS_VAR, 1, 1, 32, 32, 1>,256, 1024, 1024, 256, 1024, 1, 32, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 1024, RowParallelismTy::Low, 1, RowModTileIsZero, 2048, 2, 2, 2, K_EQUALS_VAR, 1, 1, 2, 32, 1>,1024, 2, 2, 2, 2048, 1, 2, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 512, RowParallelismTy::Low, 1, RowModTileIsZero, 4096, 8, 8, 8, K_EQUALS_VAR, 1, 1, 8, 32, 1>,512, 8, 8, 8, 4096, 1, 8, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 4096, 16, 16, 16, K_EQUALS_VAR, 1, 1, 16, 32, 1>,256, 16, 16, 16, 4096, 1, 16, 1},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 128, RowParallelismTy::Low, 1, RowModTileIsZero, 4096, 64, 64, 64, K_EQUALS_VAR, 1, 2, 16, 32, 1>,128, 64, 64, 64, 4096, 2, 16, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 128, RowParallelismTy::Low, 1, RowModTileIsZero, 8192, 64, 64, 64, K_EQUALS_VAR, 1, 4, 16, 32, 1>,128, 64, 64, 64, 8192, 4, 16, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 128, RowParallelismTy::Low, 1, RowModTileIsZero, 8192, 128, 128, 128, K_EQUALS_VAR, 1, 4, 16, 32, 1>,128, 128, 128, 128, 8192, 4, 16, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 32768, 256, 256, 64, K_EQUALS_VAR, 1, 1, 32, 32, 1>,256, 256, 256, 64, 32768, 1, 32, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 32768, 512, 512, 128, K_EQUALS_VAR, 1, 1, 32, 32, 1>,256, 512, 512, 128, 32768, 1, 32, 1},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{NULL},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 256, RowParallelismTy::Low, 1, RowModTileIsZero, 65536, 1024, 1024, 256, K_EQUALS_VAR, 1, 1, 32, 32, 1>,256, 1024, 1024, 256, 65536, 1, 32, 1}
