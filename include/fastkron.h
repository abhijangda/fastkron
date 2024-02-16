#include<cuda.h>
#include<driver_types.h>
#pragma once

enum fastKronOp {
  fastKronOp_N = 1,
  fastKronOp_T = 2
};

enum fastKronBackend {
  fastKronBackend_NONE = 0,
  fastKronBackend_X86 = 1,
  fastKronBackend_ARM = 2,
  fastKronBackend_CUDA = 3,
  fastKronBackend_ROCM = 4
};

extern "C" {
typedef struct FastKronHandle* fastKronHandle;

cudaError_t fastKronInit(fastKronHandle* handle, fastKronBackend backend, int gpus = 1, int gpusInM = -1, int gpusInK = -1, int gpuLocalKrons = -1);
void fastKronDestroy(fastKronHandle handle);

cudaError_t fastKronSetCUDAStream(fastKronHandle handle, void *ptrToStream);

//TODO: A different function for setting stream of handle 
cudaError_t gekmmSizes(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[],
                       size_t* resultSize, size_t* tempSize);

cudaError_t sgekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], float* X, 
                   fastKronOp opX, float* Fs[], fastKronOp opFs, float* Y,
                   float alpha, float beta, float *Z, float* temp1, float* temp2);
cudaError_t igekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], int* X, 
                   fastKronOp opX, int* Fs[], fastKronOp opFs, int* Y,
                   int alpha, int beta, int *Z, int* temp1, int* temp2);
cudaError_t dgekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], double* X, 
                   fastKronOp opX, double* Fs[], fastKronOp opFs, double* Y,
                   double alpha, double beta, double *Z, double* temp1, double* temp2);

cudaError_t sgekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
                       fastKronOp opX, fastKronOp opFs);
cudaError_t dgekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
                       fastKronOp opX, fastKronOp opFs);
cudaError_t igekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
                       fastKronOp opX, fastKronOp opFs);

cudaError_t kronSGEMMOutofCore(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
                               uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);
cudaError_t kronSGEMMOutofCoreX(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]);
cudaError_t kronIGEMMOutofCoreX(fastKronHandle handle, const uint NumKronMats, int* x, int* kronMats[], int** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]);

//TODO: modify such that the results are always written to the supplied result pointer 
cudaError_t kronDistributedSGEMM(fastKronHandle handle, const uint NumKronMats, float* x[], float* kronMats[], float* result[],
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], 
                                 float* temp1[], float* temp2[], cudaStream_t stream[]);

cudaError_t allocDistributedX(fastKronHandle handle, float* dX[], float* hX, uint M, uint K);
cudaError_t gatherDistributedY(fastKronHandle handle, float* dY[], float* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]);
// cudaError_t allocDistributedX(fastKronHandle handle, int* dX[], int* hX, uint M, uint K);
// cudaError_t gatherDistributedY(fastKronHandle handle, int* dY[], int* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]);
// cudaError_t allocDistributedX(fastKronHandle handle, double* dX[], double* hX, uint M, uint K);
// cudaError_t gatherDistributedY(fastKronHandle handle, double* dY[], double* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]);
}