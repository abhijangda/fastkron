# FastKron API

## Types and Enums

`enum fastKronBackend` represents backend type for FastKron with following possible values:
* `fastKronBackend_NONE`: No backend. Used as a placeholder.
* `fastKronBackend_X86`: x86 backend.
* `fastKronBackend_CUDA`: NVIDIA CUDA backend.
* `fastKronBackend_ARM`: ARM backend. **Future Work**
* `fastKronBackend_HIP`: AMD HIP backend. **Future Work**

`enum fastKronOptions` represents possible options for FastKron and has possible values:
* `fastKronOptionsNone`: No extra options and default behavior.
* `fastKronOptionsTune`: Tune for the fastest series of kernels for the given problem and use this series for subsequent calls for given problem.

`enum fastKronError` represents errors returned by FastKron API functions.
  * `fastKronSuccess`: No error. The operation was successfully executed.
  * `fastKronBackendNotAvailable`: FastKron was not compiled with the requested backend.
  * `fastKronInvalidMemoryAccess`: An invalid memory access occurred has occurred possibly because the input arrays are not of the given size.
  * `fastKronKernelNotFound`: A kernel not found for the requested problem.
  * `fastKronInvalidArgument`: An argument to the API function is invalid.
  * `fastKronInvalidKMMProblem`: Size values representing a problem are not valid.
  * `fastKronOtherError`: Undefined Error

`enum fastKronOp` represents operation on input matrices. FastKron requires all matrices to be of row major order.
* `fastKronOp_N`: No operation. The input matrix is considered as row major.
* `fastKronOp_T`: Transpose the matrix from column major to row major.

`fastKronHandle` is the type of handle for FastKron.


## Helper Functions

`fastKronError fastKronInit(fastKronHandle* handle, uint32_t backends)`

Initialize a `fastKronHandle` for one or more backends.

* **Parameters**

    * `handle`: [OUT] Pointer to a variable of `fastKronHandle`. After initialization this pointer is written.
    * `backends`: A bit-set of all backends that `fastKronHandle` can use. To use multiple backends, pass a bit-wise OR (`||`) of multiple `fastKronBackends` enums.

* **Returns**
    `fastKronSuccess` for no error or the error occurred.


`uint32_t fastKronGetBackends()`

Get a bit-wise set of all backends built in FastKron.

* **Parameters**

* **Returns**
    A bit-wise OR (`||`) of all `fastKronBackends` enum built into FastKron.

`fastKronError fastKronInitAllBackends(fastKronHandle* handle)`

Initialize a `fastKronHandle` with all backends that FastKron is compiled with.
This function has the same effect as `fastKronInit(&handle, fastKronGetBackends())`.

* **Parameters**

    * `handle`: [OUT] Pointer to a variable of `fastKronHandle`. After initialization this pointer is written.

* **Returns**
    `fastKronSuccess` for no error or the error occurred.

`fastKronError fastKronSetOptions(fastKronHandle handle, uint32_t options)`

Set one or more options to `fastKronHandle`.

* **Parameters**
    * `handle`: A variable of `fastKronHandle`.
    * `options`: A bit-wise OR (`||`) of `fastKronOptions` enum.

* **Returns**
    `fastKronSuccess` for no error or the error occurred.

`void fastKronDestroy(fastKronHandle handle)`

Destroy an initialized `fastKronHandle` handle and release all memories associated with the handle.
The handle must have been initialized before and cannot be used after without initializing it again.

* **Parameters**
    * `handle`: An initialized variable of `fastKronHandle`.

`const char* fastKronGetErrorString(fastKronError err)`

Get error description for given fastKronError.

* **Parameters**
    * `err`: A fastKronError

* **Returns**
    A null-terminated string description of error.

`fastKronError fastKronInitCUDA(fastKronHandle handle, void *ptrToStream)`

Initializes the CUDA backend with stream only if fastKronHandle was initialized with CUDA backend.

* **Parameters**
    * `handle`: A fastKronHandle initialized with CUDA backend.
    * `ptrToStream`: A pointer to CUDA stream.

* **Returns**: `fastKronSuccess` for no error or the error occurred.

`fastKronError fastKronInitHIP(fastKronHandle handle, void *ptrToStream)`

Initializes the HIP backend with stream only if fastKronHandle was initialized with CUDA backend.
**This function is not implemented yet but is provided for the future**

* **Parameters**
    * `handle`: A fastKronHandle initialized with CUDA backend.
    * `ptrToStream`: A pointer to CUDA stream.

* **Returns**: `fastKronSuccess` for no error or the error occurred.

`fastKronError fastKronInitX86(fastKronHandle handle)`

Initializes the x86 backend with stream only if fastKronHandle was initialized with x86 backend.

* **Parameters**
    * `handle`: A fastKronHandle initialized with x86 backend.

* **Returns**: `fastKronSuccess` for no error or the error occurred.

`fastKronError fastKronSetStream(fastKronHandle handle, fastKronBackend backend, void* ptrToStream)`

Set the CUDA/HIP stream for CUDA/HIP backend only if CUDA/HIP backend was initialized with the handle.

* **Parameters**
    * `handle`: A fastKronHandle initialized with CUDA or HIP backend.
    * `backend` : `fastKronBackend_CUDA` or `fastKronBackend_HIP`.
    * `ptrToStream`: A pointer to CUDA or HIP stream.

* **Returns**: `fastKronSuccess` for no error or the error occurred.


## Generalized Kronecker Matrix-Matrix Multiplication Functions 

These functions are used to do Generalized Kronecker Matrix-Matrix Multiplication (GeKMM) of the form:

$Z = \alpha ~ op(X) \times \left (op(F^1) \otimes op(F^2) \otimes \dots op(F^N) \right) + \beta Y$

where,
* $op$ is no-transpose or transpose operation on a matrix.
* each $op(F^i)$ is a row-major matrix of size $P^i \times Q^i$.
* $F^i \otimes F^j$ is Kronecker Product of two matrices
* $op(X)$ is a row-major matrix of size $M \times \left(P^1 \cdot P^2 \cdot P^3 \dots P^N \right)$
* $Y$ and $Z$ are row-major matrices of size $M \times \left(Q^1 \cdot Q^2 \cdot Q^3 \dots Q^N \right)$
* $\alpha$ and $\beta$ are scalars

`fastKronError gekmmSizes(fastKronHandle handle, 
                          uint32_t M, uint32_t N, 
                          uint32_t Ps[], uint32_t Qs[], 
                          size_t* yElems, 
                          size_t* tmpElems)`

Obtain the number of elements of the result matrix and temporary matrices for GeKMM.
The function writes to `yElems` and `tmpElems`.

* **Parameters**
    * `handle` is an initialized variable of fastKronHandle.
    * `M` is number of rows of $X$, $Y$, and $Z$.
    * `N` is number of Kronecker factors, $F^i$ s.
    * `Ps` and `Qs` are arrays containing rows and columns of all $N$ Kronecker factors.
    * `yElems` [OUT] is a pointer to the number of elements of $Y$.
    * `tmpElems` [OUT] is a pointer to the number of elements of temporary buffers required to do GeKMM.

* **Returns**
    Write values to `yElems` and `tmpElems`. Return `fastKronSuccess` for no error or the error occurred.

`fastKronError sgekmm(fastKronHandle handle, 
                      fastKronBackend backend, 
                      uint32_t M, uint32_t N,
                      uint32_t Ps[], uint32_t Qs[],
                      const float* X, fastKronOp opX,
                      const float* Fs[], fastKronOp opFs,
                      float* Z, 
                      float alpha, float beta, 
                      const float *Y,
                      float* temp1, float* temp2)`

                      
`fastKronError dgekmm(fastKronHandle handle,
                      fastKronBackend backend,
                      uint32_t M, uint32_t N,
                      uint32_t Ps[], uint32_t Qs[],
                      double* X, fastKronOp opX,
                      double* Fs[], fastKronOp opFs,
                      double* Z, 
                      double alpha, double beta,
                      double *Y,
                      double* temp1, double* temp2)`

Perform GeKMM on using 32-bit floating point or 64-bit double floating point operations on input matrices, $X$, $F^i$ s, and $Z$, and write the output to $Y$.
These functions require atleast temporary storage obtained using `gekmmSizes`. If Z and Y points to the same memory location then both temp1 and temp2 must be passed as valid memory pointers. Otherwise, only temp1 needs to be a valid memory pointer and temp2 can be NULL.
All pointers should point to either x86 CPU RAM if `backend` is x86 or NVIDIA GPU RAM if `backend` is CUDA.

* **Parameters**:
    * `handle` is an initialized variable of fastKronHandle.
    * `backend` is the `fastKronBackend` to use to perform the computation.
    * `M` is number of rows of $X$, $Y$, and $Z$.
    * `N` is the number of Kronecker factors, $F^i$ s.
    * `Ps` and `Qs` are arrays containing rows and columns of all $N$ Kronecker factors.
    * `X` is the pointer to $X$.
    * `opX` is operation on $X$ so that $op(X)$ is a row-major matrix.
    * `Fs` is an array of N pointers for each $F^i$ s.
    * `opFs` is operation on each $F^i$ so that $op(F^i)$ is a row-major matrix.
    * `Z` [OUT] is pointer to the result of GeKMM.
    * `alpha` and `beta` are the scalars
    * `Y` is pointer to $Y$. This pointer can be NULL only if `beta` is 0.
    * `temp1` is a temporary buffer required for the computation and cannot be NULL.
    * `temp2` is another temporary buffer required only when `Z` and `Y` points to the same memory location.

* **Returns**
    Write result of GeKMM to `Z`. Return `fastKronSuccess` for no error or the error occurred.