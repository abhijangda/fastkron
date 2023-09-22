#define MAX_K 4096
#define MIN_K 4096
#define MIN_KP_K 8
#define MAX_KP_K 8
#define KERNEL_DECL(T, VecT, ElemType) \
KernelInfo{(void*)kronGemmKernel<T, VecT, 512, RowParallelismTy::Low, 2, 0, 2048, 16, 8, 8, 0, 1, 2, 2, 8, 1>,512, 16, 8, 8, 2, 2048, 2, 2, 1, ElemType, 0, 0},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 512, RowParallelismTy::Low, 1, 1, 2048, 16, 8, 8, 0, 1, 1, 4, 8, 1>,512, 16, 8, 8, 1, 2048, 1, 4, 1, ElemType, 1, 0},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 512, RowParallelismTy::Low, 2, 0, 2048, 16, 8, 4, 0, 1, 1, 2, 8, 1>,512, 16, 8, 4, 2, 2048, 1, 2, 1, ElemType, 0, 0}