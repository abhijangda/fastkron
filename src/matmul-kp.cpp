#include <iostream>
#include <string>
#include <cstdlib>

void setMatrix(int* mat, int M, int N, int v) 
{
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      mat[i*N + j] = v;
    }
  }
}

void printMatrix(int* mat, int M, int N) 
{
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("v %d\n", mat[i*N + j]);
    }
  }
}


template<int NUM_KP_MATS>
void baselineKPThenMatmul(int* result, int* x, int* kpout, int* kpMats[NUM_KP_MATS],
                          int M, int N, int K, int KP_MAT_M, int KP_MAT_N, int KP_MAT_K)
{
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      kpout[i*N + j] = kpMats[0][i/KP_MAT_K * KP_MAT_N + j/KP_MAT_N] * kpMats[1][i%KP_MAT_K * KP_MAT_N + j%KP_MAT_N];
      // printf("val %d\n", kpout[i*N + j]);
    }
  }

  for(int i = 0; i < M; i++) {    
    for(int j = 0; j < N; j++) {    
      result[i* N + j] = 0;    
      for(int k = 0; k < K; k++) {   
        result[i * N + j] += x[i*K + k]*kpout[k*N + j];
      }    
    }    
  }

}

int main(int argc, char* argv[]) 
{
  const int M = 4;
  const int N = 4;
  const int K = 4;
  
  const int NUM_KP_MATS = 2;
  const int KP_MAT_M = 2;
  const int KP_MAT_N = 2;
  const int KP_MAT_K = 2;

  int *kpout = new int[K*N];
  int *x = new int[M*K];
  setMatrix(x, M, K, 1);

  int *kpMats[NUM_KP_MATS];

  for (int i = 0; i < NUM_KP_MATS; i++) {
    kpMats[i] = new int[KP_MAT_K * KP_MAT_N];
    setMatrix(kpMats[i], KP_MAT_K, KP_MAT_N, 1);
  }

  int* result = new int[M*N];

  baselineKPThenMatmul<NUM_KP_MATS>(result, x, kpout, kpMats, 
                          M, N, K, KP_MAT_M, KP_MAT_N, KP_MAT_K);
  // printMatrix(result, M, N);

  return 0;
}