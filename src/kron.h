#ifndef __KRON_H__
#define __KRON_H__

cudaError_t kronSGEMM(const uint NumKronMats, float* kronGemmResults[], float* x, float* kronMats[], float** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

cudaError_t kronIGEMM(const uint NumKronMats, int* kronGemmResults[], int* x, int* kronMats[], int** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

cudaError_t kronDGEMM(const uint NumKronMats, double* kronGemmResults[], double* x, double* kronMats[], double** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

#endif