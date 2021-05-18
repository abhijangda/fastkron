#include <iostream>
#include <string>
#include <cstdlib>
#include <cassert>

void setMatrix(int* mat, int M, int N, int (*fnvalue)()) 
{
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      mat[i*N + j] = fnvalue();
    }
  }
}

void printMatrix(int* mat, int M, int N) 
{
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("mat[%d*N + %d] = %d\n", i,j,mat[i*N + j]);
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

/**
 * 
*/
template<int NUM_KP_MATS>
void slicedMatmul(int* kpMatmulResult[NUM_KP_MATS], int* x, int* kpout, int* kpMats[NUM_KP_MATS],
                  int M, int N, int K, int KP_MAT_M, int KP_MAT_N, int KP_MAT_K)
{
  assert (M == N && N == K);
  for (int kp = 0; kp < NUM_KP_MATS; kp++) {
    int* prevKPMatmul = (kp == 0) ? x : kpMatmulResult[kp - 1];
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        int r = 0;
        for (int kp_k = 0; kp_k < KP_MAT_K; kp_k++) {
          r += prevKPMatmul[i*K + (j*KP_MAT_K)%K + kp_k] * kpMats[1][kp_k*KP_MAT_K + j % KP_MAT_N];
        }

        kpMatmulResult[kp][i*K + j] = r;
      }
    }
  }
}

bool check(int* ref, int* computed, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (ref[i*N + j] != computed[i* N + j]) {
        printf("Mismatch for %d x %d at (%d, %d): ref = %d, computed = %d\n", M, N, i, j, ref[i*N+j], computed[i*N+j]);
        return false;
      }
    }
  }

  return true;
}

int one() {return 1;}
int randMod() {return rand()%100;}

template<int NUM_KP_MATS>
void setValues(int* kpMats[NUM_KP_MATS], int *x, int M, int N, int K, int KP_MAT_M, int KP_MAT_N, int KP_MAT_K, int (*fnvalue)())
{
  for (int i = 0; i < NUM_KP_MATS; i++) {
    setMatrix(kpMats[i], KP_MAT_K, KP_MAT_N, fnvalue);
  }

  setMatrix(x, M, K, fnvalue);
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

  int *kpMats[NUM_KP_MATS];
  int* kpMatmulResult[NUM_KP_MATS];

  for (int i = 0; i < NUM_KP_MATS; i++) {
    kpMats[i] = new int[KP_MAT_K * KP_MAT_N];
    kpMatmulResult[i] = new int[M*N];
  }

  int* result = new int[M*N];
  
  int (*testCases[2])() = {&one, &randMod};

  for (int test = 0; test < sizeof(testCases)/sizeof(testCases[0]); test++) {
    setValues<NUM_KP_MATS>(kpMats, x, M, N, K, KP_MAT_M, KP_MAT_N, KP_MAT_K, testCases[test]);
    baselineKPThenMatmul<NUM_KP_MATS>(result, x, kpout, kpMats, 
                            M, N, K, KP_MAT_M, KP_MAT_N, KP_MAT_K);

    slicedMatmul<NUM_KP_MATS>(kpMatmulResult, x, kpout, kpMats,
                    M, N, K, KP_MAT_M, KP_MAT_N, KP_MAT_K);

    printf("Results Correct for test %d\n", test);
    // printMatrix(kpMatmulResult[1], M, N);
  }

  return 0;
}