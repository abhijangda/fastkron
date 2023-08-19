#define MAX_K 1024
#define MIN_K 1024
#define MIN_KP_K 2
#define MAX_KP_K 2
static uint MaxTileRowsA[] = {1, 1, 1, 1, 2, 1, 1, 1, 1, 1};

static uint MaxTileKronCols[] = {2, 4, 8, 16, 32, 64, 128, 64, 128, 256};

#define KERNEL_DECL(T, VecT, RowModTileIsZero, K_EQUALS_VAR) \
KernelInfo{(void*)kronGemmKernel<T, VecT, 512, RowParallelismTy::Low, 1, RowModTileIsZero, 1024, 2, 2, 2, K_EQUALS_VAR, 1, 1, 2, 32, 2>, 512, 2, 2, 2, 1024, 1, 2}