# FastKron

FastKron is a fast library for computing *Generalized Kronecker-Matrix Matrix Multiplication (GeKMM)* on NVIDIA GPUs and X86 CPUs.
FastKron contains a specialized algorithm and implementations of GeKMM rather than using existing linear algebra operations.
FastKron avoids extra transposes and adds more optimizations including fusion of multiple kernels.
Therefore, FastKron performs orders of magnitude better than baseline GPyTorch, NVIDIA cuTensor, and HPTT.
FastKron obtains upto 90% of the maximum FLOPs of a NVIDIA Tesla A100 and same FLOPs as Intel MKL of an AMD EPYC 7742 64-Core with AVX256.
FastKron supports float and double data type.
Fastkron provides a C++ library and a Python library compatible with PyTorch and Numpy.

For more details look [Fast Kronecker Matrix-Matrix Multiplication on GPUs](https://dl.acm.org/doi/abs/10.1145/3627535.3638489).

# Performance
We compare FastKron with state-of-the-art baselines of existing algorithms.
GPyTorch implements the traditional shuffle algorithm that uses matrix multiplication and transpose. GPyTorch runs on NVIDIA GPUs and x86 CPUs.
NVIDIA cuTensor and TCCG (https://github.com/HPAC/tccg) are tensor contraction engines for NVIDIA GPUs and x86 CPUs respectively.
Graphs below shows the performance of FastKron against these baselines.

| NVIDIA A100 SXM 80GB | AMD 7742 64-Core with AVX2|
|-|-|
| ![](https://github.com/abhijangda/fastkron/blob/main/tests/benchmarks/single-a100-flops.png?raw=true)|![](https://github.com/abhijangda/fastkron/blob/main/tests/benchmarks/single-x86-flops.png?raw=true)|

The graphs above multiplies a matrix of shape [M, P<sup>N</sup>] with a Kronecker Product of N matrices of size [P, Q].
FastKron performs significantly better than existing baselines.

# Hardware and OS Support
|  | Linux | WSL2 |
|----------|----------|----------|
| x86   | :white_check_mark:   | :white_check_mark: |
| AVX256   | :white_check_mark: | :white_check_mark: |
| AVX512   | :white_check_mark: |:white_check_mark: |
| SM50+ CUDA cores    |:white_check_mark: | :white_check_mark: |
| SM80+ Tensor cores  | :x: | :x: |

x86 CPUs older than GLIBC x86-64-v2, ARM CPUs, AMD GPUs, Windows, and Mac OS are not *yet* supported.
In these cases, the Python wrapper PyFastKron will *fallback* to the shuffle algorithm in Numpy or PyTorch.
The future plan is as follows: Windows, SM80+ Double Tensor cores, AMD GPUs, ARM CPUs.

# Example
The directory `example/` pinclude examples of using FastKron's CUDA and x86 backend using both C++ and Python.
Before using an example, follow below instructions to build FastKron. 

# Build
Build the C++ library, libFastKron.so, to use with C++ programs or the Python library, PyFastKron, to use with PyTorch or Numpy programs.

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
git clone --recurse-submodules https://github.com/abhijangda/fastkron.git
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

Run X86 CPU tests using 
```
make run-x86-tests
```

Run CUDA tests using 
```
make run-cuda-tests
```

### PyFastKron
Install PyFastKron using pip

```
pip install .
```

To disable a backend add `--config-settings=cmake.define.ENABLE_<backend>=OFF` as argument to above command.

Run tests using 
```
pytest
```

# Documentation

FastKron C++ API: [documents/api.md](https://github.com/abhijangda/FastKron/blob/main/documents/cpp-api.md)

FastKron Python API: [documents/api.md](https://github.com/abhijangda/FastKron/blob/main/documents/python-api.md)

Kernel Tuning: [documents/autotuning.md](https://github.com/abhijangda/FastKron/blob/main/documents/autotuning.md)

Multi-GPU: [documents/multigpu.md](https://github.com/abhijangda/FastKron/blob/main/documents/multigpu.md)

# Citation

```
@inproceedings{10.1145/3627535.3638489,
author = {Jangda, Abhinav and Yadav, Mohit},
title = {Fast Kronecker Matrix-Matrix Multiplication on GPUs},
year = {2024},
isbn = {9798400704352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3627535.3638489},
doi = {10.1145/3627535.3638489},
booktitle = {Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
pages = {390–403},
numpages = {14},
keywords = {graphics processing units, CUDA, kronecker product, linear algebra},
location = {Edinburgh, United Kingdom},
series = {PPoPP '24}
}
```