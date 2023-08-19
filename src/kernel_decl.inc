#define MAX_K 4096
#define MIN_K 4096
#define MIN_KP_K 8
#define MAX_KP_K 8
static uint MaxTileRowsA[] = {1, 1, 1, 1, 2, 1, 1, 1, 1, 1};

static uint MaxTileKronCols[] = {MIN_KP_K};// {2, 4, 8, 16, 32, 64, 128, 64, 128, 256};

#define KERNEL_DECL(T, VecT, RowModTileIsZero, K_EQUALS_VAR) \
KernelInfo{(void*)kronGemmKernel<T, VecT, 512, RowParallelismTy::Low, 1, RowModTileIsZero, 4096, 8, 8, 8, K_EQUALS_VAR, 1, 1, 8, 32, 3>,512, 8, 8, 8, 4096, 1, 8}


//KernelInfo{(void*)kronGemmKernel<T, VecT, 512, RowParallelismTy::Low, 1, RowModTileIsZero, 512, 8, 8, 8, K_EQUALS_VAR, 1, 1, 1, 32, 3>,512, 8, 8, 8, 512, 1, 1}