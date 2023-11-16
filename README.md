# FastKron

FastKron is an efficient and fast library for Kronecker Matrix Matrix Multiplication on both Single GPU and Multi GPUs.
FastKron performs orders of magnitude better than GPyTorch by avoiding transpose of the shuffle algorithm.
FastKron obtains upto 85% of the maximum FLOPs of Tesla V100. 
FastKron supports several datatypes: float, double, int, float.

This repository provides the source code of FastKron, Makefile, test cases, and the API.

### Build
FastKron requires generating CUDA kernels for one or more problem sizes using
`src/gen_tuner_kernels.py`. For example, generating kernels for M = 1024, N = 5, P = 8.

`python src/gen_tuner_kernels.py -same-factors 5 8,8`

Then we can build `libKron.so` using 

`make -j`

### Tests
To run tests execute

`python tests/run-tests.py`

### API
FastKron provide following API functions:
* `kron<type>GEMM` does single GPU KronMatmul, where the type follows BLAS conventions, i.e., `S` for float, `D` for double, `I` for integer, and `L` for long.
* `kronDistributed<type>GEMM ` does multi GPU KronMatmul and follows the same convention as its single-GPU counterpart
* `kron<type>Tune` performs autotuning for a given size over all compiled CUDA kernels and stores the best CUDA kernel series in its internal state.
* `fastKronInit/fastKronDestroy` initializes or destroys fastKron handle.

### Python Module
The repository contains a Python module, `PyFastKron`. The module is a Python wrapper over CUDA API functions.
It can be installed with 

`python setup.py install`

### Example
We will now show how to use FastKron in a CUDA program. The steps are:
1. Include the header file "fastKron.h".
2. Create buffers for inputs and outputs
3. Create a `fastKronHandle_t fk`
4. Initialize the handle using `fastKronInit(&fk)`
5. Tune for best kernels using `kronSGEMMTune`
6. Call KronMatmul kernel, `kronSGEMM`
7. Destroy the handle using `fastKronDestroy(fk)`
8. When compiling add the include directory and link to `libKron.so`

The `example` directory contains source code of the example.