#include "kernel_decl.inc"

#include "kernel.cuh"

#define TYPE_KERNELS(T, VecT, ElemType) \
  KERNEL_DECL(T, VecT, ElemType, 0),\
  KERNEL_DECL(T, VecT, ElemType, 1),

//Three type kernels float/float4, int/int4, and double/double4
#define NUM_TYPE_KERNELS 2
#define NUM_MAX_K_KERNELS (log2(MAX_K)-log2(MIN_K) + 1)
#define NUM_KP_N_K_KERNELS (log2(MAX_KP_K)-log2(MIN_KP_K) + 1)

#define NUM_K_EQUALS_VAR 2
#define NUM_KPK_EQUALS_VAR 1
#define NUM_ROWS_MOD_TILE_IS_ZERO 2
#define EXTERNAL_KP_K_TILE_ MAX_K

static KernelInfo KronGemmKernels[] = {
  TYPE_KERNELS(float,  float4, ElementType::Float)
  // TYPE_KERNELS(int,    int4)
  // TYPE_KERNELS(double, double4)
};