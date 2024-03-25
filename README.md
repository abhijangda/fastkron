# FastKron

FastKron is an efficient and fast library for Kronecker Matrix Matrix Multiplication on both Single GPU and Multi GPUs.
FastKron performs orders of magnitude better than GPyTorch by avoiding transpose of the shuffle algorithm.
FastKron obtains upto 85% of the maximum FLOPs of both NVIDIA Tesla V100 and NVIDIA Tesla A100. 
FastKron supports several datatypes: float, double, int, float.

This repository provides the source code of FastKron, Makefile, test cases, and the API.

### Build
FastKron requires generating CUDA kernels for one or more problem sizes using
`src/gen_tuner_kernels.py`. For example, generating CUDA kernels for M = 1024, N = 5, P = 8 with OpX and OpF set to N.

`python src/gen_tuner_kernels.py -same-factors 5 8,8 -backend cuda -opX N -opF N`

Then we can build `libFastKron.so` using 

```mkdir build/
cd build/
cmake ..
make -j
```

### Tests
To run tests execute

```
python tests/run-tests.py
```

### API
FastKron provide following API functions:
* `fastKronInit/fastKronDestroy` initializes or destroys fastKron handle.
* `<type>gekmm` does single GPU KronMatmul, where the type follows BLAS conventions, i.e., `s` for float, `d` for double, `i` for integer, and `l` for long.
* `<type>gekmmTune` performs autotuning for a given size over all compiled CUDA kernels and stores the best CUDA kernel series in its internal state.
* `kronDistributed<type>GEMM ` does multi GPU KronMatmul and follows the same convention as its single-GPU counterpart

### Python Module
The repository contains a Python module, `PyFastKron`. The module is a Python wrapper over CUDA API functions.
It can be installed with 

```
python setup.py install
```

### Example
An example CUDA program to use FastKron is written as follows:
```
//example.cu
#include <fastKron.h>

int main() {
  //Define Problem Sizes
  int N = 5;
  int M = 1024;
  int P = 8, Q = 8;
  
  //Allocate inputs and output
  float* x, *fs[], *y;
  cudaMalloc(&x, M * (int)powf(P, N) * sizeof(float));
  for (int i = 0; i < N; i++) cudaMalloc(&fs[i], P*Q * sizeof(float));
  cudaMalloc(&y, M * (int)powf(Q, N) * sizeof(float));
  
  //Initialize FastKron
  fastKronHandle handle;
  fastKronInit(&handle);
  
  //Get Temporary size and allocate temporary
  size_t tempSize;
  gekmmSizes(handle, M, N, P, Q, nullptr, &tempSize);
  float* temp;
  cudaMalloc(&temp, tempSize * sizeof(float));

  //Tune for best performing kernel
  sgekmmTune(handle, M, N, P, Q, 0);

  //Do KronMatmul using the tuned kernel
  sgekmm(handle, M, N, P, Q,  
         x, fs, y, 1, 0, nullptr, 
         temp, nullptr, 0);
  
  //Destroy FastKron
  fastKronDestroy(handle);
}
```
Compiling using nvcc, add the include directory, and link to `libFastKron.so`

```nvcc example.cu -Isrc/ -L build/ -lFastKron -o example```

Run the example 
```./example```
