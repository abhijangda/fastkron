# FastKron

FastKron is a fast library for computing *Generalized Matrix Kronecker-Matrix Multiplication (GeMKM)* and *Generalized Kronecker-Matrix Matrix Multiplication (GeKMM)* on NVIDIA GPUs and X86 CPUs.
FastKron contains specialized algorithms and implementations of GeMKM and GeKMM rather than using existing linear algebra operations.
FastKron avoids extra transposes and adds more optimizations including fusion of multiple kernels.
Therefore, FastKron performs orders of magnitude better than baseline GPyTorch, NVIDIA cuTensor, and HPTT.
Fastkron provides a C++ library and a Python library for Numpy and PyTorch autograd functions.
FastKron provides fast implementations for float and double data type, while Numpy/PyTorch functions uses Shuffle algorithm for other types.

For more details look [Fast Kronecker Matrix-Matrix Multiplication on GPUs](https://dl.acm.org/doi/abs/10.1145/3627535.3638489).

# Performance
We compare FastKron's GeMKM and GeKMM with the existing shuffle algorithm in GPyTorch based on PyTorch 2.5.1.
Below table shows the range of speedup on different hardware and data types.

### GeMKM

| Hardware | Float    | Double |
|----------|----------|--------|
| AMD 64-Core CPU with AVX| 9.3-45x| 5.8-21x|
| AMD 64-Core CPU with AVX512| 9.7-38x| 6.3-21x|
| NVIDIA A100 80 GB| 1.5-9.5x| 1.1-9.5x|
| NVIDIA V100 16 GB| 2.5-10x| 1.9-11x|

### GeKMM

| Hardware | Float    | Double |
|----------|----------|--------|
| AMD 64-Core CPU with AVX| 2.7-13.7x| 1.5-7x|
| AMD 64-Core CPU with AVX512| 2.2-14x| 2-7x|
| NVIDIA A100 80 GB|1.3-4.6x |0.9-4.5x |
| NVIDIA V100 16 GB| 1.4-6.4x|2-7.8x |

For more information see [documents/performance.md](https://github.com/abhijangda/FastKron/blob/main/documents/performance.md)

# Hardware and OS Support
|  | Linux | WSL2 | Windows | Mac |
|----------|----------|----------|-------|-----|
| x86   | ✅   | ✅ | 🐍 | 🐍 |
| ARM   | 🐍 | 🐍 | 🐍 | 🐍 |
| AVX256   | ✅ | ✅ | 🐍 | 🐍 |
| AVX512   | ✅ |✅ | 🐍 | 🐍|
| SM50+ CUDA cores    |✅ | ✅ | 🐍 | 🐍 |
| SM80+ Tensor cores  | ❌ | ❌ | 🐍 | 🐍 |
| AMD RoCM | 🐍 | 🐍 | 🐍 | 🐍 |

✅ FastKron supports optimized implementations for AVX256 and AVX512 CPUs and NVIDIA GPUs.\
❌ Tensor cores for double are not supported.\
🐍 Supported in Python module. x86 CPUs older than GLIBC x86-64-v2, ARM CPUs, AMD GPUs, Windows, and Mac OS are not supported in C++ API but PyFastKron *fallbacks* to the shuffle algorithm in Numpy or PyTorch.

The future roadmap is as follows in terms of priority: Windows, SM80+ Double Tensor cores, AMD GPUs, ARM CPUs.

# Example
The directory `example/` pinclude examples of using FastKron's CUDA and x86 backend using both C++ and Python.
Before using an example, follow below instructions to build FastKron.

# Installation

PyFastKron can be installed using pip.

```pip install pyfastkron```

PyFastKron's CUDA backend is built with CUDA 12.3 but is compatible with CUDA 11.8 and above.

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

Run tests using
```
pytest
```

# Documentation

C++ API: [documents/cpp-api.md](https://github.com/abhijangda/FastKron/blob/main/documents/cpp-api.md)\
Python API: [documents/python-api.md](https://github.com/abhijangda/FastKron/blob/main/documents/python-api.md)\
Kernel Tuning: [documents/autotuning.md](https://github.com/abhijangda/FastKron/blob/main/documents/autotuning.md)\
Performance: [documents/performance.md](https://github.com/abhijangda/FastKron/blob/main/documents/performance.md)\
Multi-GPU: [documents/multigpu.md](https://github.com/abhijangda/FastKron/blob/main/documents/multigpu.md)\
Contributing: [documents/contributing.md](https://github.com/abhijangda/FastKron/blob/main/documents/contributing.md)\

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