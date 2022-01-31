/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutass-1.3 to 
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/krongemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////


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
      // if (mat[i*N + j] == 18496)
        // printf("%d,%d\n",i,j);
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

void cutlassKronGEMM(int NUM_KP_MATS, int* kpMatmulResult[], int* x, int* kpMats[],
                     int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[])
{
  using RowMajor = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassSimt;



// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm61;
  using CutlassKronGemm = cutlass::gemm::device::KronGemm<int,        // Data-type of A matrix
                                                  RowMajor,  // Layout of A matrix
                                                  int,        // Data-type of Kron Factors matrix
                                                  RowMajor,
                                                  int,
                                                  RowMajor,
                                                  int,
                                                  MMAOp,
                                                  SmArch,
                                                  cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>>; // Layout of Kron Factors matrix

  CutlassKronGemm gemm_operator;
  
  CutlassKronGemm::TensorRefB b_krons[1];
  
  // printf("tensor_ref_x.stride() %d\n", tensor_ref_x.stride(0));
  for (int i = 0; i < NUM_KP_MATS; i++) {
    b_krons[0] = {kpMats[NUM_KP_MATS-i-1], KP_MAT_N[NUM_KP_MATS-i-1]};
    printf("kpMats[NUM_KP_MATS-i-1] %p\n", kpMats[NUM_KP_MATS-i-1]);
    CutlassKronGemm::TensorRefA tensor_ref_x = {(i==0) ? x : kpMatmulResult[i-1], M};

    CutlassKronGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                                    tensor_ref_x,    // Tensor-ref for source matrix A
                                    b_krons,
                                    1,
                                    KP_MAT_N, KP_MAT_K,
                                    {kpMatmulResult[i], M},
                                    {kpMatmulResult[i], M}); // Kron Factors

    //
    // Launch the CUTLASS GEMM kernel.
    //
    
    cutlass::Status status = gemm_operator(args);

    //
    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    //

    if (status != cutlass::Status::kSuccess) {
      printf("cutlass error\n")  ;
    }

    printf("cutlass succesfull\n");
    // Return success, if no errors were encountered.
  }
}


template<typename T, int TILE_Y, int TILE_X, int KP_N, int KP_K>
__global__ 
void cuda_gemm(int M, int N, int K, T * A, T * kron_fac, T * C) {
  /*Each threadblock compute TILE_X x KP_N of C*/

  //Each threadblock loads the KP_K x TILE_Y kron_fac into shared memory, loads every TILE_X x KP_K sub-matrix of A into shared memory,
  //multiplies each sub-matrix with every column of kron_fac, and stores the results.

  //TODO: For now TILE_Y = 1;

  __shared__ int kron_fac_sh[KP_K][TILE_Y];
  __shared__ int As[TILE_X][KP_K];
  __shared__ int Csh[TILE_X][KP_K];

  for (auto i = threadIdx.x; i < KP_K * TILE_Y; i += blockDim.x) {
    kron_fac_sh[i/TILE_Y][i%TILE_Y] = kron_fac[(i/TILE_Y) * KP_N + blockIdx.y *TILE_Y+ (i%TILE_Y)];
  }

  __syncthreads();

  int start_row = blockIdx.x * TILE_X;
  for (int a_col_batch = 0; a_col_batch < K; a_col_batch += KP_K)  {
    for (int a_row = threadIdx.x; a_row < TILE_X; a_row += blockDim.x) {
      for (int a_col = 0; a_col < KP_K; a_col++) {
        int a = A[(a_row + start_row) * K + (a_col_batch + a_col)];
        As[a_row][a_col] = a;
      }
    }
    __syncthreads();

    for (int tile_y = 0; tile_y < TILE_Y; tile_y++) {
      for (int a_row = threadIdx.x; a_row < TILE_X; a_row += blockDim.x) {
        int c = 0;

        for (int a_col = 0; a_col < KP_K; a_col++) {
          int a = As[a_row][a_col];
          int kp = kron_fac_sh[a_col][tile_y];// kron_fac[a_col * KP_K + blockIdx.y];
          // printf("%d: (%d x %d)\n", threadIdx.x, (a_row + start_row) * K + (a_col_batch + a_col), a_col * KP_K + blockIdx.y);
          c += a * kp;
        }


        Csh[a_row][a_col_batch/KP_K] = c;
      }

      __syncthreads();

      for (int a_row = threadIdx.x; a_row < TILE_X; a_row += blockDim.x) {
        int c_row = (a_row + start_row);
        int c_col = ((blockIdx.y * TILE_Y + tile_y) * KP_K + a_col_batch/KP_K);
        int c_idx = c_row * N + c_col;

        C[c_idx] = Csh[a_row][a_col_batch/KP_K];
      }
    }
  }
}

void customKronGEMM(int NUM_KP_MATS, int* kpMatmulResult[], int* x, int* kpMats[],
                     int M, int N, int K, int KP_MAT_N[], int KP_MAT_K[])
{
  //Row Major Layout of all matrics
  for (int i = 0; i < NUM_KP_MATS; i++) {
    int* prev_kp = (i==0) ? x : kpMatmulResult[i-1];
    
    const int TILE_Y = 32; //Y direction corresponds to tile of column of the KP factor
    const int TILE_X = 128; //X direction correspond to tile of row 

    dim3 grid = {M/TILE_X, (N/KP_MAT_N[NUM_KP_MATS-i-1])/TILE_Y}; 
    dim3 block = {128,1,1};
    cuda_gemm<int,TILE_Y,TILE_X,32,32><<<grid, block>>>(M, N, K, prev_kp, kpMats[NUM_KP_MATS-i-1], kpMatmulResult[i]);

    // CUDACHECK(cudaDeviceSynchronize());
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

///////////////////////////////////////////////////////////////////////////////////////////////////

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
  std::vector<MatrixSizes> matrixSizes = {
                                          // {4,4,4, 2, {2,2},{2,2}},
                                          // {4,4,6, 2, {1,4},{2,3}},
                                          // {4,4,8, 2, {2,2},{2,4}},
                                          // {4,4,8, 2, {2,2},{4,2}},
                                          // {8,8,8, 2, {4,2},{4,2}},
                                          // {8,8,8, 2, {4,2},{2,4}},
                                          // {8,8,8, 3, {2,2,2},{2,2,2}},
                                          // {8,8,32, 3, {2,2,2},{2,4,4}},
                                          // {8,16,32, 3, {4,2,2},{2,4,4}},
                                          // {8,8,16, 3, {2,2,2},{2,4,2}},
                                          // {16,8,8, 3, {2,2,2},{2,2,2}},
                                          // {16,16,16, 2, {4,4},{4,4}},
                                          // {16,16,16, 3, {4,2,2},{4,2,2}},
                                          // {16,16,16, 3, {4,2,2},{2,4,2}},
                                          // {16,16,16, 3, {8,2,1},{2,4,2}},
                                          // {16,16,16, 4, {2,2,2,2},{2,2,2,2}},
                                          // {16,16,64, 4, {2,2,2,2},{2,4,2,4}},
                                          // {256,256,256, 4, {4,4,4,4},{4,4,4,4}},
                                          // {256,256,256, 2, {16,16},{16,16}},
  #ifdef EVAL
                                          {65536,1024,1024, 2, {32,32},{32,32}},
  #else
                                          {512,1024,1024, 2, {32,32},{32,32}},
  #endif

                                          // {1024, 1024, 1024, 2, {32,32},{32,32}}
                                          };

  // int (*fnvalues[4])(int, int) = {&one, &zeroOne, &setToI, &randMod};
  int (*fnvalues[1])(int, int) = {&randMod};

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
    int *kpMats[NUM_KP_MATS];
    int* kpMatmulResult[NUM_KP_MATS];

    int *x = new int[M*K];

    int* dX;
    int** dKpOut;
    int** dKpMats;
    int** dKpMatmulResult;
    
    CUDACHECK(cudaMalloc(&dX, M*K * sizeof(int)));
    CUDACHECK(cudaMalloc(&dKpMats, NUM_KP_MATS * sizeof(int*)));
    CUDACHECK(cudaMalloc(&dKpMatmulResult, NUM_KP_MATS * sizeof(int*)));
    CUDACHECK(cudaMalloc(&dKpOut, NUM_KP_MATS * sizeof(int*)));

    int* __dKpOut[NUM_KP_MATS];
    int* __dKpMats[NUM_KP_MATS];
    int* __dKpMatmulResult[NUM_KP_MATS];

    for (int i = 0; i < NUM_KP_MATS; i++) {
      KP_MAT_K[i] = matrixSize.KP_MAT_K[i];
      KP_MAT_N[i] = matrixSize.KP_MAT_N[i];
      kpMats[i] = new int[KP_MAT_K[i] * KP_MAT_N[i]];
      kpout[i] = new int[K*N]; //TODO: larger than needed
      kpMatmulResult[i] = new int[M*std::max(N,K)];

      CUDACHECK(cudaMalloc(&__dKpMats[i], KP_MAT_K[i] * KP_MAT_N[i] * sizeof(int)));
      // CUDACHECK(cudaMalloc(&__dKpOut[i], K * N * sizeof(int)));
      CUDACHECK(cudaMalloc(&__dKpMatmulResult[i], M*std::max(N,K) * sizeof(int)));

      CUDACHECK(cudaMemset(__dKpMatmulResult[i], 0, M*std::max(N,K) * sizeof(int)));
      // CUDACHECK(cudaMemset(__dKpOut[i], 0, K * N * sizeof(int)));
    }

    // CUDACHECK(cudaMemcpy(&dKpOut[0], &__dKpOut[0], NUM_KP_MATS * sizeof(int*), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(&dKpMats[0], &__dKpMats[0], NUM_KP_MATS * sizeof(int*), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(&dKpMatmulResult[0], &__dKpMatmulResult[0], NUM_KP_MATS * sizeof(int*), cudaMemcpyHostToDevice));

    int* result = new int[M*N];

    int* dResult;

    CUDACHECK(cudaMalloc(&dResult, M * N * sizeof(int)));

    for (int fnvalue = 0; fnvalue < sizeof(fnvalues)/sizeof(fnvalues[0]); fnvalue++) {
      setValues(NUM_KP_MATS, kpMats, x, M, N, K, KP_MAT_N, KP_MAT_K, fnvalues[fnvalue]);

      for (int i = 0; i < NUM_KP_MATS; i++) {
        CUDACHECK(cudaMemcpy(__dKpMats[i], kpMats[i], KP_MAT_K[i] * KP_MAT_N[i] * sizeof(int), cudaMemcpyHostToDevice));
      }
    
      CUDACHECK(cudaMemcpy(dX, x, M * K * sizeof(int), cudaMemcpyHostToDevice));
  #ifndef EVAL
      baselineKPThenMatmul(NUM_KP_MATS, result, x, kpout, kpMats, 
                           M, N, K, KP_MAT_N, KP_MAT_K);
  #endif
      // slicedMatmul(NUM_KP_MATS, kpMatmulResult, x, kpMats,
      //              M, N, K, KP_MAT_N, KP_MAT_K);

      for (int i = 0; i < NUM_KP_MATS; i++)
        CUDACHECK(cudaMemset(__dKpMatmulResult[i], 0, M*std::max(N,K) * sizeof(int)));
  #ifdef EVAL
      for (int i = 0; i < 100; i++)
        customKronGEMM(NUM_KP_MATS, __dKpMatmulResult, dX, __dKpMats, M, N, K, KP_MAT_N, KP_MAT_K);
      CUDACHECK(cudaDeviceSynchronize());
      return;
  #else
      for (int i = 0; i < 1; i++)
        customKronGEMM(NUM_KP_MATS, __dKpMatmulResult, dX, __dKpMats, M, N, K, KP_MAT_N, KP_MAT_K);
  #endif
      CUDACHECK(cudaDeviceSynchronize());
      // return;
      int* hKpMatMulResult = new int[M*N];
      // return;
      for (int i = 0; i < NUM_KP_MATS; i++)
        CUDACHECK(cudaMemcpy(kpMatmulResult[i], __dKpMatmulResult[i], M*N*sizeof(int), cudaMemcpyDeviceToHost));
      // if (check(result, kpMatmulResult[NUM_KP_MATS-1], M, N))
      if (check(result, kpMatmulResult[NUM_KP_MATS-1], M,N))
        printf("Results Correct for test %d\n", fnvalue);
      else {
        // printf("\nMatmul:");
        // printMatrix(result, K, N);

        // printf("\nx:");
        // printMatrix(x, M, K);    
        // for (int kpMatId = 0; kpMatId < NUM_KP_MATS; kpMatId++) {
        //   printf("\nKP Mat %d:", kpMatId);
        //   printMatrix(kpMats[kpMatId], KP_MAT_K[kpMatId], KP_MAT_N[kpMatId]);
        // }
        // // printf("\nKP Out:");
        // // printMatrix(kpout[0], 8, 8);
        // for (int id = 0; id < NUM_KP_MATS; id++) {
        //   printf("\nKP result %d:", id);
        //   printMatrix(kpMatmulResult[id], M, N);
        // }
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