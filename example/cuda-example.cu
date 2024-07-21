//cuda-example.cu
#include <fastkron.h>

int main() {
  //Define Problem Sizes
  uint32_t N = 5;
  uint32_t M = 1024;
  uint32_t Ps[5] = {8,8,8,8,8}, Qs[5] = {8,8,8,8,8};

  //Allocate inputs and output
  float* x, *fs[N], *y;
  cudaMalloc(&x, M * (int)powf(Ps[0], N) * sizeof(float));
  for (int i = 0; i < N; i++) cudaMalloc(&fs[i], Ps[0]*Qs[0] * sizeof(float));
  cudaMalloc(&y, M * (int)powf(Qs[0], N) * sizeof(float));
  
  //Initialize FastKron with CUDA
  fastKronHandle handle;
  fastKronInit(&handle, fastKronBackend_CUDA);
  
  //Initialize FastKron's CUDA with stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  fastKronInitCUDA(handle, (void*)&stream);

  //Get Temporary size and allocate temporary
  size_t tempSize, resultSize;
  gekmmSizes(handle, M, N, Ps, Qs, &resultSize, &tempSize);

  float* temp;
  cudaMalloc(&temp, tempSize * sizeof(float));

  //Do KronMatmul using the tuned kernel

  sgekmm(handle, fastKronBackend_CUDA, M, N, Ps, Qs,
         x, fastKronOp_N, fs, fastKronOp_N, y, 1, 0, nullptr, 
         temp, nullptr);
  
  //Destroy FastKron
  fastKronDestroy(handle);
}