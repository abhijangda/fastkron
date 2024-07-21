//x86-example.cpp
#include <fastkron.h>

#include <math.h>

int main() {
  //Define Problem Sizes
  uint32_t N = 5;
  uint32_t M = 1024;
  uint32_t Ps[5] = {8,8,8,8,8}, Qs[5] = {8,8,8,8,8};

  //Allocate inputs and output
  float* x, *fs[N], *y;
  x = new float[M * (int)powf(Ps[0], N)];
  for (int i = 0; i < N; i++) fs[i] = new float[Ps[0]*Qs[0]];
  y = new float[M * (int)powf(Qs[0], N)];
  
  //Initialize FastKron with all backends (CUDA and x86 by default)
  fastKronHandle handle;
  fastKronInitAllBackends(&handle);

  //Get Temporary size and allocate temporary
  size_t tempSize, resultSize;
  gekmmSizes(handle, M, N, Ps, Qs, &resultSize, &tempSize);

  float* temp;
  temp = new float[tempSize];

  //Do KronMatmul
  sgekmm(handle, fastKronBackend_X86, M, N, Ps, Qs,
         x, fastKronOp_N, fs, fastKronOp_N, y, 1, 0, nullptr, 
         temp, nullptr);
  
  //Destroy FastKron
  fastKronDestroy(handle);
}