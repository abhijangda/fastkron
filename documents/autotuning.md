# Tune FastKron for a GeKMM Problem
 
The space of all valid x86/CUDA/HIP/ARM kernels to run a computation can contain 1000s of kernels.
Hence, it is not practical to building all kernels for each possible kernel size and ship these kernels as part of FastKron.
Instead, FastKron contains a set of few efficient pre-selected kernels and at runtime select the fastest series of kernels to compute the problem. 

FastKron contains three modes to select the best kernel series:
* *Online Selection* uses an algorithm that selects the fastest series of kernel. The algorithm balances between cache size (shared memory) and parallelism. However, the algorithm is not always correct in finding the fastest series.
* *Fast Tuning* runs all pre-selected kernels for the problem and selects the fastest kernel series. This tuning is done only once for the problem and the selected kernel series is called for subsequent execution for same problem sizes, therefore, amortizing the cost of tuning.
* *Full Tuning* generates all valid kernels for a given problem size, builds FastKron library, runs all kernels to find the fastest kernel series, and use these kernels for subsequent execution of same problem size. Similar to Fast Tuning, Full Tuning is done only once for the problem size, thus, amortizing the cost of tuning.

Lets now see how to use all three modes.

#### Online Selection

By default FastKron uses Online Selection algorithm to select the fastest kernel series.

#### Fast Tuning

Setting `fastKronOptionsTune` as an option using `fastKronSetOptions` enables Fast Tuning.
This option must be set before calling any of the `*gekmm` functions.

#### Full Tuning

