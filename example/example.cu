//example.cu
#include <fastkron.h>

int main() {
  //Define Problem Sizes
  int N = 5;
  int M = 1024;
  int P = 8, Q = 8;
  uint Ps[N], Qs[N];

  //Allocate inputs and output
  float* x, *fs[N], *y;
  cudaMalloc(&x, M * (int)powf(P, N) * sizeof(float));
  for (int i = 0; i < N; i++) {
    cudaMalloc(&fs[i], P*Q * sizeof(float));
    Ps[i] = P;
    Qs[i] = Q;
  }
  cudaMalloc(&y, M * (int)powf(Q, N) * sizeof(float));
  
  //Initialize FastKron
  fastKronHandle handle;
  fastKronInit(&handle);
  
  //Get Temporary size and allocate temporary
  size_t tempSize;
  gekmmSizes(handle, M, N, Ps, Qs, nullptr, &tempSize);
  float* temp;
  cudaMalloc(&temp, tempSize * sizeof(float));

  //Tune for best performing kernel
  sgekmmTune(handle, M, N, Ps, Qs, 0);

  //Do KronMatmul using the tuned kernel
  sgekmm(handle, M, N, Ps, Qs,  
         x, fs, y, 1, 0, nullptr, 
         temp, nullptr, 0);
  
  //Destroy FastKron
  fastKronDestroy(handle);
  return 0;
}