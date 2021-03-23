#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IDX2F(i,j,ld) ((i*ld)+j)


// suppose you need to compute Kronecker Product of A and B, i.e., C = A x B then.
// C[0: len(B)]  = saxpy(A[0][0], B).
// C[len(B) : len(B) * 2] = saxply(A[0][1], B).

// This function multiplies the vector x by the scalar α and adds it to the vector y
// overwriting the latest vector with the result. Hence, the performed operation is
// y [ j ] = α × x [ k ] + y [ j ] for i = 1 , … , n , k = 1 + ( i - 1 ) *  incx and j = 1 + ( i - 1 ) *  incy .
// Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.
// cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
//                            const float           *alpha,
//                            const float           *x, int incx,
//                            float                 *y, int incy);
//
//
//
 
void debugMatrix(float *A, int N){
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			std::cout << A[IDX2F(i,j,N)] <<" ";
		}
		std::cout << "\n";
	}
}


int main(int argc, char *argv []){
	int N = 3;
	int M = 3;
	
	float * A = (float *)malloc (N * N * sizeof (float));
	float * B = (float *)malloc (M * M * sizeof (float));
	float * res = (float *)malloc(M * N *M *N * sizeof(float));
	
	float *A_dev,*B_dev,* R_dev;
	
	cudaError_t cudaStat;
    	cublasStatus_t stat;
	cublasHandle_t handle;


    	if (!A) {
        	printf ("host memory allocation failed");
        	return EXIT_FAILURE;
    	}
    	for (int i = 0; i < N; i++) {
        	for (int j = 0; j < N; j++) {
            		A[IDX2F(i,j,N)] = (i+1)*j;
        	}
    	}
	for (int i = 0; i < M; i++){
		for(int j=0; j< M; j++){
			B[IDX2F(i,j,M)] = (i+1)*j;
		}
	} 
	debugMatrix(A,N);
	debugMatrix(B,N);


    	cudaStat = cudaMalloc ((void**)&A_dev, N*N*sizeof(float));
	cudaStat = cudaMemcpy(A_dev,A, N*N*sizeof(float), cudaMemcpyHostToDevice);
   	if (cudaStat != cudaSuccess) {
        	printf ("device memory allocation failed");
        	return EXIT_FAILURE;
    	}
	cudaStat = cudaMalloc ((void**)&B_dev, M*M*sizeof(float));
	cudaStat = cudaMemcpy(B_dev,A, M*M*sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                return EXIT_FAILURE;
        }

	cudaStat = cudaMalloc((void **)&R_dev, M * N * M * N * sizeof(float));
		
    	
	stat = cublasCreate(&handle);
   	if (stat != CUBLAS_STATUS_SUCCESS) {
        	printf ("CUBLAS initialization failed\n");
        	return EXIT_FAILURE;
    	}

	float *alpha_dev;
	cudaMalloc((void **)&alpha_dev, sizeof(float));
	
	for(int i=0; i < N; i++){
		for(int j=0; j< N; j++){
			for(int k=0;k<N;k++){
			cublasSaxpy(handle, M, 
	                            &A[IDX2F(i,j,N)], 
        	                    &B_dev[IDX2F(k,0,M)], 1,
                		    &R_dev[IDX2F(i*N,j*N,M*N)], 1);

			cudaDeviceSynchronize();
			}
		}		
	}
	cudaMemcpy(res,R_dev,M*N*M*N*sizeof(float), cudaMemcpyDeviceToHost);
	debugMatrix(res,M*N);
	std::cout << "Hello World!";
   	return 0;
}
