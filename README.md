# FastKron

FastKron is a fast library for computing Generalized Kronecker-Matrix Matrix Multiplication (GeKMM) on NVIDIA GPUs and X86 CPUs.
FastKron contains a specialized algorithm and implementations of GeKMM rather than using existing linear algebra operations.
FastKron avoids extra transposes and adds more optimizations including fusion of multiple kernels.
Therefore, FastKron performs orders of magnitude better than baseline GPyTorch, NVIDIA cuTensor, and HPTT.
FastKron obtains upto 90% of the maximum FLOPs of a NVIDIA Tesla A100 and XX% of an AMD EPYC XXXX 64-Core with AVX512. 
FastKron supports float and double data type.
Fastkron provides a C++ library and a Python library compatible with PyTorch and Numpy.

# Add Graphs

# Hardware Support Matrix
|  | Linux | WSL2 |
|----------|----------|----------|
| x86  SIMD   | :white_check_mark:   | :white_check_mark: |
| AVX256   | :white_check_mark: | :white_check_mark: |
| AVX512   | :white_check_mark: |:white_check_mark: |
| SM50+ CUDA cores    |:white_check_mark: | :white_check_mark: |
| SM80+ Tensor cores  | :x: | :x: |

# Example Usage


# Build from scratch
Build the C++ library, libFastKron.so, to use with C++ programs or the Python library PyFastKron.  

### Required Pre-requisites
On Ubuntu :
```
sudo apt update && sudo apt install gcc linux-headers-$(uname -r) make g++ git python3-dev wget unzip python3-pip build-essential devscripts debhelper fakeroot intel-mkl cmake
```

### CUDA Pre-requisite
Install CUDA 11+ from https://developer.nvidia.com/cuda/ .

### Clone repository
Clone repository with submodules using 
```
git clone --recurse-submodules https://github.com/parasailteam/cusync.git
```

If already cloned and want to only clone submodules, use
```
git submodule update --init --recursive
```

### libFastKron
Build FastKron as C++ library using below commands: 

```mkdir build/
cd build/
cmake ..
make -j
```

To install run
```make install```

By default both x86 and CUDA backends are built. use CMAKE option `-DENABLE_CUDA=OFF` to disable CUDA backend or `-DENABLE_X86=OFF` to disable x86 backend.

#### Run Tests

Run tests using 
```
make tests
```

### PyFastKron
Install PyFastKron using pip

```
pip install .
```

#### Run Tests
Run Python tests using pytest

```
pytest
```

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
