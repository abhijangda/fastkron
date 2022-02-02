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
        int v2 = kpMats[kp+1][(i%KP_MAT_K[kp+1]) * KP_MAT_N[kp+1] + j%KP_MAT_N[kp+1]];
        int v1 = kpFirst[(i/KP_MAT_K[kp+1]) * kpFirstCols + j/KP_MAT_N[kp+1]];
        kpout[kp][i*cols + j] = v1 * v2;
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
  int secFacRowMulSize = 1;
  int rowsTillNow = 1;
  int colsTillNow = 1;
  int resultCols;
  for (int kp = 0; kp < NUM_KP_MATS; kp++) {
    int* prevKPMatmul = (kp == 0) ? x : kpMatmulResult[kp - 1];
    int kpSecondK = KP_MAT_K[NUM_KP_MATS - 1 - kp];
    int kpSecondN = KP_MAT_N[NUM_KP_MATS - 1 - kp];
    int prevKPMatmulCols = (kp == 0) ? K : resultCols;

    resultCols = (prevKPMatmulCols/kpSecondK) * kpSecondN;
    secFacRowMulSize = (kp == 0) ? K/kpSecondK : rowsTillNow * K/(colsTillNow * KP_MAT_K[NUM_KP_MATS - 1 - (kp)]);

    //Number of times a column is multiplied with input matrix is equal to 
    //N/(number of column elements of this matrix * cols so far) * number of rows so far.

    rowsTillNow *= KP_MAT_N[NUM_KP_MATS - 1 - (kp)];
    colsTillNow *= KP_MAT_K[NUM_KP_MATS - 1 - (kp)];

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < resultCols; j++) {
        int r = 0;

        for (int kp_k = 0; kp_k < kpSecondK; kp_k++) {
          int slice = (j / secFacRowMulSize) % kpSecondN;

          int v2 = kpMats[NUM_KP_MATS - 1 - kp][kp_k*kpSecondN + slice];
          
          r += prevKPMatmul[i* prevKPMatmulCols + (j*kpSecondK)%prevKPMatmulCols + kp_k] * v2;
        }

        kpMatmulResult[kp][i*resultCols + j] = r;
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
                                          {4,4,6, 2, {1,4},{2,3}},
                                          {4,4,8, 2, {2,2},{2,4}},
                                          {4,4,8, 2, {2,2},{4,2}},
                                          {8,8,8, 2, {4,2},{4,2}},
                                          {8,8,8, 2, {4,2},{2,4}},
                                          {8,8,8, 3, {2,2,2},{2,2,2}},
                                          {8,8,32, 3, {2,2,2},{2,4,4}},
                                          {8,16,32, 3, {4,2,2},{2,4,4}},
                                          {8,8,16, 3, {2,2,2},{2,4,2}},
                                          {16,8,8, 3, {2,2,2},{2,2,2}},
                                          {16,16,16, 2, {4,4},{4,4}},
                                          {16,16,16, 3, {4,2,2},{4,2,2}},
                                          {16,16,16, 3, {4,2,2},{2,4,2}},
                                          {16,16,16, 3, {8,2,1},{2,4,2}},
                                          {16,16,16, 4, {2,2,2,2},{2,2,2,2}},
                                          {16,16,64, 4, {2,2,2,2},{2,4,2,4}},
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
    int n=1,k=1;
    for (int i = 0; i < NUM_KP_MATS; i++) {
      k *= matrixSize.KP_MAT_K[i];
      n *= matrixSize.KP_MAT_N[i];
    }
    if (n != N || k != K) {
      printf("Invalid KP Factors Sizes %d != %d, %d != %d\n", n, N, k, K);
    }

    int *kpout[NUM_KP_MATS];
    int *x = new int[M*K];

    int *kpMats[NUM_KP_MATS];
    int* kpMatmulResult[NUM_KP_MATS];

    for (int i = 0; i < NUM_KP_MATS; i++) {
      KP_MAT_K[i] = matrixSize.KP_MAT_K[i];
      KP_MAT_N[i] = matrixSize.KP_MAT_N[i];
      kpMats[i] = new int[KP_MAT_K[i] * KP_MAT_N[i]];
      kpout[i] = new int[K*N]; //TODO: larger than needed
      kpMatmulResult[i] = new int[M*std::max(N,K)];
    }

    int* result = new int[M*N];
    

    for (int fnvalue = 0; fnvalue < sizeof(fnvalues)/sizeof(fnvalues[0]); fnvalue++) {
      setValues(NUM_KP_MATS, kpMats, x, M, N, K, KP_MAT_N, KP_MAT_K, fnvalues[fnvalue]);
      baselineKPThenMatmul(NUM_KP_MATS, result, x, kpout, kpMats, 
                              M, N, K, KP_MAT_N, KP_MAT_K);

      slicedMatmul(NUM_KP_MATS, kpMatmulResult, x, kpMats,
                      M, N, K, KP_MAT_N, KP_MAT_K);
      if (check(result, kpMatmulResult[NUM_KP_MATS-1], M, N))
        printf("Results Correct for test %d\n", fnvalue);
      else {
        // printf("\nMatmul:");
        // printMatrix(result, K, N);

        printf("\nx:");
        printMatrix(x, M, K);    
        for (int kpMatId = 0; kpMatId < NUM_KP_MATS; kpMatId++) {
          printf("\nKP Mat %d:", kpMatId);
          printMatrix(kpMats[kpMatId], KP_MAT_K[kpMatId], KP_MAT_N[kpMatId]);
        }
        // printf("\nKP Out:");
        // printMatrix(kpout[0], 8, 8);
        for (int id = 0; id < NUM_KP_MATS; id++) {
          printf("\nKP result %d:", id);
          printMatrix(kpMatmulResult[id], M, N);
        }
        // printf("\nKP result 2:");
        // printMatrix(kpMatmulResult[2], 16, 16);
        // printf("\nKP result 3:");
        // printMatrix(kpMatmulResult[3], 16, 16);
        // printf("\nKP result 1:");
        // printMatrix(kpMatmulResult[1], M, N);
        // printf("\n");
        return 0;
      }
    }

    //Is there really a need to free anything when you have tons of RAM, am I right?
  }

  return 0;
}