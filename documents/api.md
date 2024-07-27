# FastKron API

### Types and Enums

`enum fastKronBackend` represents backend type for FastKron with following possible values:
* `  fastKronBackend_NONE`: No backend. Used as a placeholder.
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

`fastKronHandle` is the type of handle for FastKron.


### API Functions

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

fastKronError fastKronInitHIP(fastKronHandle handle, void *ptrToStream);
fastKronError fastKronInitX86(fastKronHandle handle);
fastKronError fastKronSetStream(fastKronHandle handle, fastKronBackend backend, void* ptrToStream);

//TODO: A different function for setting stream of handle
fastKronError gekmmSizes(fastKronHandle handle, uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                       size_t* resultSize, size_t* tempSize);

fastKronError sgekmm(fastKronHandle handle, fastKronBackend backend, 
                     uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                     float* X, fastKronOp opX,
                     float* Fs[], fastKronOp opFs,
                     float* Y, float alpha, float beta,
                     float *Z, float* temp1, float* temp2);
fastKronError igekmm(fastKronHandle handle, fastKronBackend backend,
                     uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                     int* X, fastKronOp opX,
                     int* Fs[], fastKronOp opFs, 
                     int* Y, int alpha, int beta,
                     int *Z, int* temp1, int* temp2);
fastKronError dgekmm(fastKronHandle handle, fastKronBackend backend,
                     uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                     double* X, fastKronOp opX,
                     double* Fs[], fastKronOp opFs,
                     double* Y, double alpha, double beta,
                     double *Z, double* temp1, double* temp2);