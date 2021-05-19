#include <iostream>
#include <string>
#include <cstdlib>
#include <cassert>
#include <vector>

void setMatrix(int* mat, int M, int N, int (*fnvalue)(int i, int j)) 
{
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      mat[i*N + j] = fnvalue(i,j);
    }
  }
}

void printMatrix(int* mat, int M, int N) 
{
  printf("[");
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%d, ", mat[i*N + j]);
    }
    if (i < M-1)
      printf("\n");
  }
  printf("]");
}

void baselineKPThenMatmul(int NUM_KP_MATS, int* result, int* x, int* kpout[], int* kpMats[],
                          int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[])
{
  int cols;
  int rows;

  for (int kp = 0; kp < NUM_KP_MATS - 1; kp++) {
    int* kpFirst = (kp == 0) ? kpMats[0] : kpout[kp - 1];
    int kpFirstRows = (kp == 0) ? KP_MAT_K[0] : rows;
    int kpFirstCols = (kp == 0) ? KP_MAT_N[0] : cols;

    cols = kpFirstCols * KP_MAT_N[kp+1];
    rows = kpFirstRows * KP_MAT_K[kp+1];

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        kpout[kp][i*cols + j] = kpFirst[(i/kpFirstRows) * kpFirstCols + j/kpFirstCols] * kpMats[kp+1][i%KP_MAT_K[kp+1] * KP_MAT_N[kp+1] + j%KP_MAT_N[kp+1]];
        // printf("val %d\n", kpout[i*N + j]);
      }
    }
  }

  for(int i = 0; i < M; i++) {    
    for(int j = 0; j < N; j++) {    
      result[i* N + j] = 0;    
      for(int k = 0; k < K; k++) {   
        result[i * N + j] += x[i*K + k]*kpout[NUM_KP_MATS-2][k*N + j];
      }    
    }    
  }
}

/**
 * 
*/
void slicedMatmul(int NUM_KP_MATS, int* kpMatmulResult[], int* x, int* kpMats[],
                  int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[])
{
  for (int kp = 0; kp < NUM_KP_MATS; kp++) {
    int* prevKPMatmul = (kp == 0) ? x : kpMatmulResult[kp - 1];
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        int r = 0;
        for (int kp_k = 0; kp_k < KP_MAT_K[kp]; kp_k++) {
          r += prevKPMatmul[i*K + (j*KP_MAT_K[kp])%K + kp_k] * kpMats[NUM_KP_MATS - 1 - kp][kp_k*KP_MAT_K[kp] + j / KP_MAT_N[kp]];
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

int one(int i, int j) {return 1;}
int zeroOne(int i, int j) {return i % 2;}
int setToI(int i, int j) {return i;}
int randMod(int i, int j) {return rand()%10;}

void setValues(int NUM_KP_MATS, int* kpMats[], int *x, int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[], int (*fnvalue)(int i, int j))
{
  for (int i = 0; i < NUM_KP_MATS; i++) {
    setMatrix(kpMats[i], KP_MAT_K[i], KP_MAT_N[i], fnvalue);
  }

  setMatrix(x, M, K, fnvalue);
}

struct MatrixSizes {
  const int M, N, K;
  const int NUM_KP_MATS;
  const std::vector<int> KP_MAT_N; 
  const std::vector<int> KP_MAT_K;
};

int main(int argc, char* argv[]) 
{
  std::vector<MatrixSizes> matrixSizes = {{4,4,4, 2, {2,2},{2,2}},
                                          {256,256,256, 2, {16,16},{16,16}},
                                          {256,256,256, 4, {4,4,4,4},{4,4,4,4}},
                                          {1024,1024,1024, 2, {32,32},{32,32}}
                                          };

  int (*fnvalues[4])(int, int) = {&one, &zeroOne, &setToI, &randMod};

  for (MatrixSizes matrixSize : matrixSizes) {
    int M = matrixSize.M;
    int N = matrixSize.N;
    int K = matrixSize.K;
    
    int NUM_KP_MATS = matrixSize.NUM_KP_MATS;
    int KP_MAT_N[NUM_KP_MATS];
    int KP_MAT_K[NUM_KP_MATS];

    printf("Matmul: %d x %d x %d, Num KP Factors: %d\n", M, N, K, NUM_KP_MATS);
    int *kpout[NUM_KP_MATS-1];
    int *x = new int[M*K];

    int *kpMats[NUM_KP_MATS];
    int* kpMatmulResult[NUM_KP_MATS];

    for (int i = 0; i < NUM_KP_MATS; i++) {
      KP_MAT_K[i] = matrixSize.KP_MAT_K[i];
      KP_MAT_N[i] = matrixSize.KP_MAT_N[i];
      kpMats[i] = new int[KP_MAT_K[i] * KP_MAT_N[i]];
      kpout[i] = new int[K*N]; //TODO: larger than needed
      kpMatmulResult[i] = new int[M*N];
    }

    int* result = new int[M*N];
    

    for (int fnvalue = 0; fnvalue < sizeof(fnvalues)/sizeof(fnvalues[0]); fnvalue++) {
      setValues(NUM_KP_MATS, kpMats, x, M, N, K, KP_MAT_N, KP_MAT_K, fnvalues[fnvalue]);
      baselineKPThenMatmul(NUM_KP_MATS, result, x, kpout, kpMats, 
                              M, N, K, KP_MAT_N, KP_MAT_K);

      slicedMatmul(NUM_KP_MATS, kpMatmulResult, x, kpMats,
                      M, N, K, KP_MAT_N, KP_MAT_K);
      if (check(result, kpMatmulResult[1], M, N))
        printf("Results Correct for test %d\n", fnvalue);
      else {
        // printf("x:");
        // printMatrix(x, M, K);    
        // printf("\nA:");
        // printMatrix(kpMats[0], KP_MAT_K[0], KP_MAT_N[0]);
        // printf("\nB:");  
        // printMatrix(kpMats[1], KP_MAT_K[1], KP_MAT_N[1]);
        // printf("\nKP Out:");
        // printMatrix(kpout[0], K, N);
        // printf("\nKP result 0:");
        // printMatrix(kpMatmulResult[0], M, N);
        // printf("\nKP result 1:");
        // printMatrix(kpMatmulResult[1], M, N);
        // printf("\n");
        return 0;
      }
    }

    //Why is there a need to free anything, am I right?
  }

  return 0;
}