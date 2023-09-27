#define MAX_K 1048576
#define MIN_K 1048576
#define MIN_KP_K 32
#define MAX_KP_K 32
#define KERNEL_DECL(T, VecT, ElemType) \
KernelInfo{(void*)kronGemmKernel<T, VecT, 128, RowParallelismTy::Low, 2, 1, 4096, 64, 64, 64, 0, 1, 2, 16, 32, 1, 0>,128, 64, 64, 64, 2, 4096, 2, 16, 1, ElemType, 1, 0, 0},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 128, RowParallelismTy::Low, 2, 1, 4096, 64, 64, 64, 0, 1, 2, 16, 32, 1, 1>,128, 64, 64, 64, 2, 4096, 2, 16, 1, ElemType, 1, 0, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 128, RowParallelismTy::Low, 1, 1, 8192, 128, 128, 128, 0, 1, 2, 32, 32, 1, 1>,128, 128, 128, 128, 1, 8192, 2, 32, 1, ElemType, 1, 0, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 128, RowParallelismTy::Low, 1, 1, 8192, 128, 128, 128, 0, 1, 2, 32, 32, 1, 0>,128, 128, 128, 128, 1, 8192, 2, 32, 1, ElemType, 1, 0, 0}