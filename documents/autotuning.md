# Kernel Tuning for a GeKMM Problem
 
The space of all valid x86/CUDA/HIP/ARM kernels to run a computation can contain 1000s of kernels.
Hence, it is not practical to building all kernels for each possible kernel size and ship these kernels as part of FastKron.
Instead, FastKron contains a set of few efficient pre-selected kernels and at runtime select the fastest series of kernels to compute the problem. 

FastKron contains three modes to select the best kernel series:
* *Online Selection* uses an algorithm that selects the fastest series of kernel. The algorithm balances between cache size (shared memory) and parallelism. However, the algorithm is not always correct in finding the fastest series.
* *Fast Tuning* runs all pre-selected kernels for the problem and selects the fastest kernel series. This tuning is done only once for the problem and the selected kernel series is called for subsequent execution for same problem sizes, therefore, amortizing the cost of tuning.
* *Full Tuning* generates all valid kernels for a given problem size, builds FastKron library, runs all kernels to find the fastest kernel series, and use these kernels for subsequent execution of same problem size. Similar to Fast Tuning, Full Tuning is done only once for the problem size, thus, amortizing the cost of tuning.

Lets see how to use all three modes.

#### Online Selection

By default FastKron uses Online Selection algorithm to select the fastest kernel series.

#### Fast Tuning

Setting `fastKronOptionsTune` as an option using `fastKronSetOptions` enables Fast Tuning.
This option must be set before calling any of the `*gekmm` functions.

#### Full Tuning

Suppose the GeKMM problem is:

$Z = \alpha ~ op(X) \times \left (op(F^1) \otimes op(F^2) \otimes \dots op(F^N) \right) + \beta Y$

where,
* $op$ is no-transpose or transpose operation on a matrix.
* each $op(F^i)$ is a row-major matrix of size $P^i \times Q^i$.
* $F^i \otimes F^j$ is Kronecker Product of two matrices
* $op(X)$ is a row-major matrix of size $M \times \left(P^1 \cdot P^2 \cdot P^3 \dots P^N \right)$
* $Y$ and $Z$ are row-major matrices of size $M \times \left(Q^1 \cdot Q^2 \cdot Q^3 \dots Q^N \right)$
* $\alpha$ and $\beta$ are scalars

The first step is to generate all valid kernels for the problem sizes using `src/gen_tuner_kernels.py`.

```
python ../src/gen_tuner_kernels.py -backend <x86 or cuda> -archs <x86 or cuda archs> -distinct-factors N P1,Q1 P2,Q2 P3,Q3 ... -types <float or double> -opt-levels 3
```

For example, generate CUDA kernels for Ampere architecture (SM80+) with N=4 and all Ps = Qs = 8.

```
python ../src/gen_tuner_kernels.py -backend cuda -archs ampere -distinct-factors 3 8,8 8,8 8,8 -types float -opt-levels 3 
```

The next step is to run CMake with `FULL_TUNE=ON` and enable the backend but switch off other backends, and do make.

```
mkdir build/
cd build/
cmake .. -DFULL_TUNE=ON -DENABLE_<backend>=ON
make -j
```

The final step is to use `fastKronSetOptions()` to set `fastKronOptionsTune`.