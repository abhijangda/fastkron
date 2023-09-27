#define MAX_K 16777216
#define MIN_K 16777216
#define MIN_KP_K 16
#define MAX_KP_K 16
#define KERNEL_DECL(T, VecT, ElemType) \
KernelInfo{(void*)kronGemmKernel<T, VecT, 128, RowParallelismTy::Low, 1, 1, 8192, 16, 16, 16, 0, 1, 4, 16, 16, 1, 0>,128, 16, 16, 16, 1, 8192, 4, 16, 1, ElemType, 1, 0, 0}

/*
KernelInfo{(void*)kronGemmKernel<T, VecT, 128, RowParallelismTy::Low, 2, 1, 2048, 16, 16, 8, 0, 1, 2, 4, 16, 1, 1>,128, 16, 16, 8, 2, 2048, 2, 4, 1, ElemType, 1, 0, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 64, RowParallelismTy::Low, 2, 0, 64, 16, 16, 16, 0, 1, 1, 1, 16, 1, 1>,64, 16, 16, 16, 2, 64, 1, 1, 1, ElemType, 0, 0, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 128, RowParallelismTy::Low, 2, 0, 1024, 16, 16, 8, 0, 1, 4, 1, 16, 1, 0>,128, 16, 16, 8, 2, 1024, 4, 1, 1, ElemType, 0, 0, 0},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 32, RowParallelismTy::Low, 2, 1, 512, 16, 16, 16, 0, 1, 1, 16, 16, 1, 0>,32, 16, 16, 16, 2, 512, 1, 16, 1, ElemType, 1, 0, 0},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 64, RowParallelismTy::Low, 2, 1, 1024, 16, 16, 4, 0, 1, 4, 1, 16, 1, 0>,64, 16, 16, 4, 2, 1024, 4, 1, 1, ElemType, 1, 0, 0},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 32, RowParallelismTy::Low, 2, 0, 256, 16, 16, 8, 0, 1, 4, 1, 16, 1, 0>,32, 16, 16, 8, 2, 256, 4, 1, 1, ElemType, 0, 0, 0},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 32, RowParallelismTy::Low, 2, 0, 256, 16, 16, 4, 0, 1, 2, 1, 16, 1, 0>,32, 16, 16, 4, 2, 256, 2, 1, 1, ElemType, 0, 0, 0},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 32, RowParallelismTy::Low, 1, 1, 512, 16, 16, 4, 0, 1, 2, 2, 16, 1, 1>,32, 16, 16, 4, 1, 512, 2, 2, 1, ElemType, 1, 0, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 64, RowParallelismTy::Low, 1, 1, 4096, 16, 16, 4, 0, 1, 4, 4, 16, 1, 1>,64, 16, 16, 4, 1, 4096, 4, 4, 1, ElemType, 1, 0, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 128, RowParallelismTy::Low, 1, 1, 1024, 16, 16, 8, 0, 1, 2, 2, 16, 1, 1>,128, 16, 16, 8, 1, 1024, 2, 2, 1, ElemType, 1, 0, 1},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 512, RowParallelismTy::Low, 2, 1, 4096, 16, 16, 16, 0, 1, 4, 2, 16, 1, 0>,512, 16, 16, 16, 2, 4096, 4, 2, 1, ElemType, 1, 0, 0},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 64, RowParallelismTy::Low, 2, 0, 256, 16, 16, 8, 0, 1, 1, 2, 16, 1, 0>,64, 16, 16, 8, 2, 256, 1, 2, 1, ElemType, 0, 0, 0},\
KernelInfo{(void*)kronGemmKernel<T, VecT, 32, RowParallelismTy::Low, 1, 1, 2048, 16, 16, 8, 0, 1, 4, 8, 16, 1, 1>,32, 16, 16, 8, 1, 2048, 4, 8, 1, ElemType, 1, 0, 1}*/