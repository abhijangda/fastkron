#ifndef __KRON_H__
#define __KRON_H__

float* kronSGEMM(const int NUM_KP_MATS, float* kpMatmulResult[], float* x, float* kpMats[],
  int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[], cudaStream_t stream);

int* kronIGEMM(const int NUM_KP_MATS, int* kpMatmulResult[], int* x, int* kpMats[],
  int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[], cudaStream_t stream);

#endif