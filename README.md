# FastKron

FastKron is an efficient and fast library for Kronecker Matrix Matrix Multiplication on both Single GPU and Multi GPUs.
FastKron performs orders of magnitude better than GPyTorch by avoiding transpose of the shuffle algorithm.
FastKron obtains upto 85% of the maximum FLOPs of both NVIDIA Tesla V100 and NVIDIA Tesla A100. 
FastKron supports several datatypes: float, double, int, float.

This repository provides the source code of FastKron, Makefile, test cases, and the API.

### Build
FastKron requires generating CUDA kernels for one or more problem sizes using
`src/gen_tuner_kernels.py`. For example, generating kernels for M = 1024, N = 5, P = 8.

`python src/gen_tuner_kernels.py -same-factors 5 8,8`

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
`example/` contains an example file to perform KronMatmul using FastKron.