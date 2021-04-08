#include <chrono>
#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define IDX2F(i,j,ld) (((i)*(ld))+(j))


using namespace std::chrono;

// suppose you need to compute Kronecker Product of A and B, i.e., C = A x B then.
// Launch blocks equal to N*N
// Blocks equal N*N, threads N*N 
// debug, then scale to larger matrix
// add code for shared memory. 
__global__
void kron_prod(float *A, float *B, float *R, int N,int M){
	int i = blockIdx.x;
	int a_r = i/M;
	int b_r = i%M;
	int b_c = threadIdx.x;
	for(int a_c=0;a_c<N;a_c++){
		R[IDX2F(i,(a_c*M)+b_c,N*M)]=A[IDX2F(a_r,a_c,N)] * B[IDX2F(b_r,b_c,M)];
	}

}

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
	int M = 4;

	if(argc==3){
		 N = std::stoi(argv[1]);
	 	 M = std::stoi(argv[2]);
	}
	
	float * A = (float *)malloc (N * N * sizeof (float));
	float * B = (float *)malloc (M * M * sizeof (float));
	float * res = (float *)malloc(M * N *M *N * sizeof(float));
	
	float *A_dev,*B_dev,* R_dev;
	
	cudaError_t cudaStat;

    	if (!A) {
        	printf ("host memory allocation failed");
        	return EXIT_FAILURE;
    	}
    	for (int i = 0; i < N; i++) {
        	for (int j = 0; j < N; j++) {
            		A[IDX2F(i,j,N)] = (i)*j+1;
        	}
    	}
	for (int i = 0; i < M; i++){
		for(int j=0; j< M; j++){
			B[IDX2F(i,j,M)] = 1;
			//B[IDX2F(i,j,M)] = (i)*j+1;
		}
	} 
	
	debugMatrix(A,N);
	debugMatrix(B,M);
	//std::cout <<"############\n";

    	cudaStat = cudaMalloc ((void**)&A_dev, N*N*sizeof(float));
	cudaStat = cudaMemcpy(A_dev,A, N*N*sizeof(float), cudaMemcpyHostToDevice);
   	if (cudaStat != cudaSuccess) {
        	printf ("device memory allocation failed");
        	return EXIT_FAILURE;
    	}
	cudaStat = cudaMalloc ((void**)&B_dev, M*M*sizeof(float));
	cudaStat = cudaMemcpy(B_dev,B, M*M*sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                return EXIT_FAILURE;
        }

	cudaStat = cudaMalloc((void **)&R_dev, M * N * M * N * sizeof(float));
   	memset(res,0,sizeof(float)*M*N*M*N);		
    
	float *alpha_dev;
	cudaMalloc((void **)&alpha_dev, sizeof(float));

	cudaDeviceSynchronize();

	auto start = high_resolution_clock::now();
	for(int loop=0;loop<1;loop++){
		kron_prod<<<N*M,M>>>(A_dev,B_dev,R_dev,N,M);
	}
	cudaDeviceSynchronize();
	auto stop = high_resolution_clock::now();
	// add a test for the returned result
	cudaMemcpy(res,R_dev,M*N*M*N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	auto duration = duration_cast<milliseconds>(stop - start);
  
    	std::cout << "Time taken by saxpy: "
         << duration.count() << " milli" << std::endl;

	debugMatrix(res,M*N);
   	return 0;
}
