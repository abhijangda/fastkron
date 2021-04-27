#include <chrono>
#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#define IDX2F(i,j,ld) (((i)*(ld))+(j))


using namespace std::chrono;

void debugMatrix(float *A, int N){
        for(int i=0; i<N; i++){
                for(int j=0; j<N; j++){
                        std::cout << A[IDX2F(i,j,N)] <<" ";
                }
                std::cout << "\n";
        }
}

// inputs are addresses of first element in row. Do not use IDX2F in this function 
void multiply_row(float *A_row, int N, float *B_row, int M, float *res_row){
	// non simd code 
	if (false){
		for(int i=0;i<N;i++){
			for(int j=0;j<M;j++){
				res_row[((i*M)+j)] = A_row[i] * B_row[j];	
		
			}
		}
	}else{
		 for(int i=0;i<N;i++){
			 __m256 a_val =  _mm256_set1_ps(A_row[i]);
			 for(int j=0;j<M;j=j+8){
				//res_row[((i*M)+j)] = A_row[i] * B_row[j];
				__m256 b_val =  _mm256_load_ps(&B_row[j]);
				__m256 t = _mm256_mul_ps(a_val,b_val);
			        _mm256_stream_ps(&res_row[((i*M)+j)],t);	
                        }
                }
	}

}


void kronfun(float *A, int N, float *B, int M, float *res){
	assert(N%2==0);
	for(int i=0;i<N;i=i+2){
		float *A_row = &A[i*N];
		for(int j=0;j<M;j++){
			float *B_row = &B[j*M];
			multiply_row(A_row, N, B_row, M, &res[IDX2F(i*M+j,0,M*N)]);
		}
		
		A_row = &A[(i+1)*N];
		for(int j=M-1;j>=0;j--){
			float *B_row = &B[j*M];
			multiply_row(A_row,N,B_row,M, &res[IDX2F((i+1)*M+j,0,M*N)]);
		}
	}
}


int main(int argc, char *argv []){
	int N = 16;
	int M = 16;

	if(argc==3){
		 N = std::stoi(argv[1]);
	 	 M = std::stoi(argv[2]);
	}
	
	float * A = (float *)aligned_alloc (32,N * N * sizeof (float));
	float * B = (float *)aligned_alloc (32,M * M * sizeof (float));
	float * res = (float *)aligned_alloc(32,M * N *M *N * sizeof(float));
	
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

  	
	kronfun(A, N, B, M, res);
	
	debugMatrix(res, M*N);
	std::cout << "Hello world!!\n";
   	return 0;
}
