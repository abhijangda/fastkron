#include<cuda.h>
#include<driver_types.h>
#pragma once

extern "C" {
typedef struct FastKronHandle* fastKronHandle;

cudaError_t fastKronInit(fastKronHandle* handle, int gpus = 1, int gpusInM = -1, int gpusInK = -1, int gpuLocalKrons = -1);
void fastKronDestroy(fastKronHandle handle);
cudaError_t gekmmSizes(fastKronHandle handle, const uint NumKronMats, uint M, uint N, uint K, 
                       uint KronMatCols[], uint KronMatRows[], size_t* resultSize, size_t* tempSize);

cudaError_t sgekmm(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], float* result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float* temp1, float* temp2, 
                      float alpha, float beta, float *z, cudaStream_t stream);

cudaError_t igekmm(fastKronHandle handle, const uint NumKronMats, int* x, int* kronMats[], int* result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], int* temp1, int* temp2, 
                      int alpha, int beta, int *z, cudaStream_t stream);

cudaError_t dgekmm(fastKronHandle handle, const uint NumKronMats, double* x, double* kronMats[], double* result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], double* temp1, double* temp2, 
                      double alpha, double beta, double *z, cudaStream_t stream);

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

cudaError_t sgekmmTune(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], 
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream);
cudaError_t dgekmmTune(fastKronHandle handle, const uint NumKronMats, double* x, double* kronMats[], 
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream);
cudaError_t idgemmTune(fastKronHandle handle, const uint NumKronMats, int* x, int* kronMats[],
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream);
}