#include <stdint.h>
#include <cstddef>

#pragma once

#define FastKronCHECK(cmd) do {                        \
  fastKronError e = cmd;                              \
  if(e != fastKronSuccess) {      \
    printf("Failed: FastKron error %s:%d at %s:%d \n",       \
        fastKronGetErrorString(e),e,__FILE__,__LINE__);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)                                          \


/****************
  Types and Enums
 ****************/

/**
 * enum fastKronOp - represents operation on input matrices.
 * FastKron requires all matrices to be of row major order.
 * @fastKronOp_N: No operation.
                  The input matrix is considered as row major.
 * @fastKronOp_T: Transpose the matrix from column major to row major.
 */
enum fastKronOp {
  fastKronOp_N = 1,
  fastKronOp_T = 2
};

/**
 * enum fastKronBackend - represents backend type 
 * for FastKron with following possible values:
 * @fastKronBackend_NONE : No backend. Used as a placeholder.
 * @fastKronBackend_X86  : x86 backend.
 * @fastKronBackend_ARM  : ARM backend. **Future Work**
 * @fastKronBackend_CUDA : NVIDIA CUDA backend.
 * @fastKronBackend_HIP  : AMD HIP backend. **Future Work**
 */
enum fastKronBackend {
  fastKronBackend_NONE = 1 << 0,
  fastKronBackend_X86  = 1 << 1,
  fastKronBackend_ARM  = 1 << 2,
  fastKronBackend_CUDA = 1 << 3,
  fastKronBackend_HIP  = 1 << 4
};

/**
 * enum fastKronOptions - represents possible options for 
 * FastKron and has possible values:
 * @fastKronOptionsNone      : No extra options and default behavior.
 * @fastKronOptionsUseFusion : Avoid selecting fused kernels.
 * @fastKronOptionsTune      : Tune for the fastest series of kernels for the given problem and
 *                             use this series for subsequent calls for given problem
 */
enum fastKronOptions {
  fastKronOptionsNone      = 1 << 0,
  fastKronOptionsUseFusion = 1 << 1,
  fastKronOptionsTune      = 1 << 2,
};

/**
 * enum fastKronError - represents errors returned by FastKron API functions.
 * @fastKronSuccess             : No error. The operation was successfully executed.
 * @fastKronBackendNotAvailable : FastKron not compiled with requested backend
 * @fastKronInvalidMemoryAccess : An invalid memory access occurred has occurred possibly 
 *                                because the input arrays are not of the given size.
 * @fastKronKernelNotFound      : A kernel not found for the requested problem.
 * @fastKronInvalidArgument     : An argument to the API function is invalid.
 * @fastKronInvalidKMMProblem   : Size values representing a problem are not valid.
 * @fastKronOtherError          : Undefined Error
 */
enum fastKronError {
  fastKronSuccess             = 0,
  fastKronBackendNotAvailable = 1,
  fastKronInvalidMemoryAccess = 2,
  fastKronKernelNotFound      = 3,
  fastKronInvalidArgument     = 4,
  fastKronInvalidKMMProblem   = 5,
  fastKronOtherError          = 6,
};

extern "C" {
typedef void* fastKronHandle;

/******************
  Helper Functions
 ******************/
/**
 * fastKronVersion() - Get FastKron version.
 *
 * Return: A string constant representing the version. 
 */
const char* fastKronVersion();
/**
 * fastKronCUDAArchs() - Get CUDA architectures supported by FastKron.
 *
 * Return:
 */
const char* fastKronCUDAArchs();

/**
 * fastKronGetErrorString() - Get error description for given fastKronError.
 * @err: A fastKronError
 *
 * Return: A null-terminated string description of error.
 */
const char* fastKronGetErrorString(fastKronError err);

/**
 * fastKronGetBackends() - Get a bit-wise set of all backends built in FastKron.
 *
 * Return: A bit-wise OR (`||`) of all `fastKronBackends` enum built into FastKron.
 */
uint32_t fastKronGetBackends();

/**
 * fastKronInit() - Initialize a `fastKronHandle` for one or more backends.
 * @handle: [OUT] Pointer to a variable of `fastKronHandle`. After initialization this pointer is written.
 * @backends: A bit-set of all backends that `fastKronHandle` can use. 
              To use multiple backends, pass a bit-wise OR (`||`) of multiple 
              `fastKronBackends` enums.
 * Return: `fastKronSuccess` for no error or the error occurred.
 */
fastKronError fastKronInit(fastKronHandle* handle, uint32_t backends);

/**
 * fastKronInitAllBackends() - Initialize a `fastKronHandle` with all backends that FastKron is compiled with.
 * @handle: [OUT] Pointer to a variable of `fastKronHandle`. 
            After initialization this pointer is written.
 *
 * This function has the same effect as `fastKronInit(&handle, fastKronGetBackends())`.

 * Return: `fastKronSuccess` for no error or the error occurred.
 */
fastKronError fastKronInitAllBackends(fastKronHandle* handle);

/**
 * fastKronSetOptions() - Set one or more options to `fastKronHandle`.
 * @handle: An initialized object of `fastKronHandle`.
 * @options: A bit-wise OR (`||`) of `fastKronOptions` enum.

 * Return: `fastKronSuccess` for no error or the error occurred.
 */
fastKronError fastKronSetOptions(fastKronHandle handle, uint32_t options);

/**
 * fastKronDestroy() - Destroy an initialized `fastKronHandle` handle.
 * @handle: An initialized variable of `fastKronHandle`.

  Destroy an initialized `fastKronHandle` handle and release all memories associated 
  with the handle. The handle must have been initialized before and cannot be used after 
  without initializing it again.
  
  * Return: 
 */
void fastKronDestroy(fastKronHandle handle);

/**
 * fastKronInitCUDA() - Initializes the CUDA backend with stream.
 * @handle: A fastKronHandle initialized with CUDA backend.
 * @ptrToStream: A pointer to the CUDA stream.
 *
 * Initializes the CUDA backend with stream only if fastKronHandle was 
 * initialized with CUDA backend.
 * 
 * Return: `fastKronSuccess` for no error or the error occurred.
 */
fastKronError fastKronInitCUDA(fastKronHandle handle, void *ptrToStream);

/** 
 * fastKronInitHIP() - Initializes the HIP backend with stream.
 * @handle: A fastKronHandle initialized with HIP backend.
 * @ptrTostream: A pointer to HIP stream.
 *
 * Initializes the HIP backend with stream only if fastKronHandle was 
 * initialized with HIP backend.
 * This function is not implemented yet but is provided for the future
 *
 * Return: `fastKronSuccess` for no error or the error occurred.
*/
fastKronError fastKronInitHIP(fastKronHandle handle, void *ptrToStream);

/**
 * fastKronInitX86() - Initializes the x86 backend with stream.
 * @handle: A fastKronHandle initialized with x86 backend.
 *
 * Initializes the x86 backend with stream only if fastKronHandle was 
 * initialized with x86 backend.
 *
 * Return: `fastKronSuccess` for no error or the error occurred.
 */
fastKronError fastKronInitX86(fastKronHandle handle);

/**
 * fastKronSetStream() - Set the CUDA/HIP stream for CUDA/HIP backend only if CUDA/HIP backend was initialized with the handle.
 * @handle: A fastKronHandle initialized with CUDA or HIP backend.
 * @backend: `fastKronBackend_CUDA` or `fastKronBackend_HIP`.
 * @ptrToStream: A pointer to CUDA or HIP stream.
 *
 * Return: `fastKronSuccess` for no error or the error occurred.
 */
fastKronError fastKronSetStream(fastKronHandle handle, fastKronBackend backend, void* ptrToStream);

/**
 * Generalized Kronecker Matrix Multiplication fuctions
 * These functions are used to do two kinds of Generalized Kronecker Matrix-Matrix Multiplication (GeKMM).
 * First is Matrix-Kronecker Matrix (MKM) Multiplication of the form:
 * 
 * $Z = \alpha ~ op(X) \times \left (op(F^1) \otimes op(F^2) \otimes \dots op(F^N) \right) + \beta Y$
 *
 * where,
  * $op$ is no-transpose or transpose operation on a matrix.
  * each $op(F^i)$ is a row-major matrix of size $P^i \times Q^i$.
  * $F^i \otimes F^j$ is Kronecker Product of two matrices
  * $op(X)$ is a row-major matrix of size $M \times \left(P^1 \cdot P^2 \cdot P^3 \dots P^N \right)$
  * $Y$ and $Z$ are row-major matrices of size $M \times \left(Q^1 \cdot Q^2 \cdot Q^3 \dots Q^N \right)$
  * $\alpha$ and $\beta$ are scalars
 * 
 * Second is Kronecker Matrix-Matrix (KMM) Multiplication of the form:
 * 
 * $Z = \alpha ~ \left (op(F^1) \otimes op(F^2) \otimes \dots op(F^N) \right) \times op(X) + \beta Y$
 *
 * where,
  * $op$ is no-transpose or transpose operation on a matrix.
  * each $op(F^i)$ is a row-major matrix of size $Q^i \times P^i$.
  * $F^i \otimes F^j$ is Kronecker Product of two matrices
  * $op(X)$ is a row-major matrix of size $\left(P^1 \cdot P^2 \cdot P^3 \dots P^N \right) \times M$
  * $Y$ and $Z$ are row-major matrices of size $\left(Q^1 \cdot Q^2 \cdot Q^3 \dots Q^N \right) \times M$
  * $\alpha$ and $\beta$ are scalars
*/


/**
 * gekmmSizes() - Obtain the number of elements of the result matrix and temporary matrices for GeKMM.
 *                The function writes to `yElems` and `tmpElems`.
 * @handle: is an initialized variable of fastKronHandle.
 * @M: is number of rows of $X$, $Y$, and $Z$.
 * @N: is number of Kronecker factors, $F^i$ s.
 * @Ps: is an array containing rows of all N Kronecker factors.
 * @Qs: is an array containing columns of all N Kronecker factors.
 * @yElems: [OUT] is a pointer to the number of elements of $Y$.
 * tmpElems: [OUT] is a pointer to the number of elements of temporary buffers required to do GeKMM.
 *
 * Returns: Return `fastKronSuccess` for no error or the error occurred.
 *          Write values to `yElems` and `tmpElems`.
 */
fastKronError gekmmSizes(fastKronHandle handle, uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                         size_t* yElems, size_t* tmpElems);
fastKronError gekmmSizesForward(fastKronHandle handle, uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                                size_t* resultSize, size_t* intermediateSizes);

/**
 * sgekmm(), igekmm(), dgekmm() - Perform MKM.
 * @handle: is an initialized variable of fastKronHandle.
 * @backend: is the `fastKronBackend` to use to perform the computation.
 * @M: is number of rows of $X$, $Y$, and $Z$.
 * @N: is the number of Kronecker factors, $F^i$ s.
 * @Ps: is an array containing rows of all N Kronecker factors.
 * @Qs: is an array containing columns of all N Kronecker factors.
 * @X: is the pointer to $X$.
 * @opX: is operation on $X$ so that $op(X)$ is a row-major matrix.
 * @Fs: is an array of N pointers for each $F^i$ s.
 * @opFs: is operation on each $F^i$ so that $op(F^i)$ is a row-major matrix.
 * @Z: [OUT] is pointer to the result of GeKMM.
 * @alpha: scalar 
 * @beta: scalar
 * @Y: is pointer to $Y$. This pointer can be NULL only if `beta` is 0.
 * @temp1: is a temporary buffer required for the computation and cannot be NULL.
 * @temp2: is another temporary buffer required only when `Z` and `Y` points to the same memory location.

 * Perform GeKMM using 32-bit floating point or 64-bit double floating point operations on 
 * input matrices, $X$, $F^i$ s, and $Z$, and write the output to $Y$. These functions 
 * require atleast temporary storage obtained using `gekmmSizes`. If Z and Y points 
 * to the same memory location then both temp1 and temp2 must be passed as valid 
 * memory pointers. Otherwise, only temp1 needs to be a valid memory pointer and 
 * temp2 can be NULL. All pointers should point to either x86 CPU RAM if 
 * `backend` is x86 or NVIDIA GPU RAM if `backend` is CUDA.

 * Return: Write result of GeKMM to `Z`. Return `fastKronSuccess` for no error
   or the error occurred.
 */
fastKronError sgemkm(fastKronHandle handle, fastKronBackend backend, 
                     uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                     const float* X, fastKronOp opX,
                     const float* const Fs[], fastKronOp opFs,
                     float* Z, float alpha, float beta,
                     const float *Y, float* temp1, float* temp2);

fastKronError dgemkm(fastKronHandle handle, fastKronBackend backend,
                     uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                     const double* X, fastKronOp opX,
                     const double* const Fs[], fastKronOp opFs,
                     double* Z, double alpha, double beta,
                     const double *Y, double* temp1, double* temp2);

/**
 * sgekmm(), igekmm(), dgekmm() - Perform GeKMM.
 * @handle: is an initialized variable of fastKronHandle.
 * @backend: is the `fastKronBackend` to use to perform the computation.
 * @M: is number of rows of $X$, $Y$, and $Z$.
 * @Qs: is an array containing columns of all N Kronecker factors.
 * @Ps: is an array containing rows of all N Kronecker factors.
 * @N: is the number of Kronecker factors, $F^i$ s.
 * @Fs: is an array of N pointers for each $F^i$ s.
 * @opFs: is operation on each $F^i$ so that $op(F^i)$ is a row-major matrix.
 * @X: is the pointer to $X$.
 * @opX: is operation on $X$ so that $op(X)$ is a row-major matrix.
 * @Z: [OUT] is pointer to the result of GeKMM.
 * @alpha: scalar 
 * @beta: scalar
 * @Y: is pointer to $Y$. This pointer can be NULL only if `beta` is 0.
 * @temp1: is a temporary buffer required for the computation and cannot be NULL.
 * @temp2: is another temporary buffer required only when `Z` and `Y` points to the same memory location.

 * Perform GeKMM using 32-bit floating point or 64-bit double floating point operations on 
 * input matrices, $X$, $F^i$ s, and $Z$, and write the output to $Y$. These functions 
 * require atleast temporary storage obtained using `gekmmSizes`. If Z and Y points 
 * to the same memory location then both temp1 and temp2 must be passed as valid 
 * memory pointers. Otherwise, only temp1 needs to be a valid memory pointer and 
 * temp2 can be NULL. All pointers should point to either x86 CPU RAM if 
 * `backend` is x86 or NVIDIA GPU RAM if `backend` is CUDA.

 * Return: Write result of GeKMM to `Z`. Return `fastKronSuccess` for no error
   or the error occurred.
 */
fastKronError sgekmm(fastKronHandle handle, fastKronBackend backend, 
                     uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                     const float* const Fs[], fastKronOp opFs,
                     const float* X, fastKronOp opX,
                     float* Z, float alpha, float beta,
                     const float *Y, float* temp1, float* temp2);

fastKronError dgekmm(fastKronHandle handle, fastKronBackend backend, 
                     uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                     const double* const Fs[], fastKronOp opFs,
                     const double* X, fastKronOp opX,
                     double* Z, double alpha, double beta,
                     const double *Y, double* temp1, double* temp2);

/**
 * Strided Batched Generalized Kronecker Matrix Multiplication functions
 * These functions perform batched MKM and KMM. These functions take extra arguments for batch count
 * of Z, and stride for batch of X, all Fs, Y, and Z.
 */

/**
 * sgemkmStridedBatched, igemkmStridedBatched, dgemkmStridedBatched - Strided Batched MKM
 * @handle: is an initialized variable of fastKronHandle.
 * @backend: is the `fastKronBackend` to use to perform the computation.
 * @M: is number of rows of $X$, $Y$, and $Z$.
 * @N: is the number of Kronecker factors, $F^i$ s.
 * @Ps: is an array containing rows of all N Kronecker factors.
 * @Qs: is an array containing columns of all N Kronecker factors.
 * @X: is the pointer to $X$.
 * @opX: is operation on $X$ so that $op(X)$ is a row-major matrix.
 * @strideX: stride of each batch for $X$.
 * @Fs: is an array of N pointers for each $F^i$ s.
 * @opFs: is operation on each $F^i$ so that $op(F^i)$ is a row-major matrix.
 * @strideF: is stride of batch for each factor.
 * @Z: [OUT] is pointer to the result of GeKMM.
 * @strideZ: stride of batch of Z.
 * @alpha: scalar 
 * @beta: scalar
 * @Y: is pointer to $Y$. This pointer can be NULL only if `beta` is 0.
 * @strideY: stride of batch of Y.
 * @temp1: is a temporary buffer required for the computation and cannot be NULL.
 * @temp2: is another temporary buffer required only when `Z` and `Y` points to the same memory location.

 * Perform GeKMM using 32-bit floating point or 64-bit double floating point operations on 
 * input matrices, $X$, $F^i$ s, and $Z$, and write the output to $Y$. These functions 
 * require atleast temporary storage obtained using `gekmmSizes`. If Z and Y points 
 * to the same memory location then both temp1 and temp2 must be passed as valid 
 * memory pointers. Otherwise, only temp1 needs to be a valid memory pointer and 
 * temp2 can be NULL. All pointers should point to either x86 CPU RAM if 
 * `backend` is x86 or NVIDIA GPU RAM if `backend` is CUDA.

 * Return: Write result of GeKMM to `Z`. Return `fastKronSuccess` for no error
   or the error occurred.
 */
fastKronError sgemkmStridedBatched(fastKronHandle handle, fastKronBackend backend, 
                                   uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                                   const float* X, fastKronOp opX, uint64_t strideX,
                                   const float* const Fs[], fastKronOp opFs, uint64_t strideF[],
                                   float* Z, uint64_t strideZ, float alpha, float beta,
                                   uint32_t batchCount, const float *Y, uint64_t strideY, 
                                   float* temp1, float* temp2);

fastKronError dgemkmStridedBatched(fastKronHandle handle, fastKronBackend backend,
                                   uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                                   const double* X, fastKronOp opX, uint64_t strideX,
                                   const double* const Fs[], fastKronOp opFs, uint64_t strideF[], 
                                   double* Z, uint64_t strideZ, double alpha, double beta,
                                   uint32_t batchCount, const double *Y, uint64_t strideY,
                                   double* temp1, double* temp2);

/**
 * sgekmmStridedBatched, igekmmStridedBatched, dgekmmStridedBatched - Strided Batched KMM
 * @handle: is an initialized variable of fastKronHandle.
 * @backend: is the `fastKronBackend` to use to perform the computation.
 * @N: is the number of Kronecker factors, $F^i$ s.
 * @Qs: is an array containing columns of all N Kronecker factors.
 * @Ps: is an array containing rows of all N Kronecker factors.
 * @M: is number of rows of $X$, $Y$, and $Z$.
 * @Fs: is an array of N pointers for each $F^i$ s.
 * @opFs: is operation on each $F^i$ so that $op(F^i)$ is a row-major matrix.
 * @strideF: is stride of batch for each factor.
 * @X: is the pointer to $X$.
 * @opX: is operation on $X$ so that $op(X)$ is a row-major matrix.
 * @strideX: stride of each batch for $X$.
 * @Z: [OUT] is pointer to the result of GeKMM.
 * @strideZ: stride of batch of Z.
 * @alpha: scalar 
 * @beta: scalar
 * @Y: is pointer to $Y$. This pointer can be NULL only if `beta` is 0.
 * @strideY: stride of batch of Y.
 * @temp1: is a temporary buffer required for the computation and cannot be NULL.
 * @temp2: is another temporary buffer required only when `Z` and `Y` points to the same memory location.

 * Perform GeKMM using 32-bit floating point or 64-bit double floating point operations on 
 * input matrices, $X$, $F^i$ s, and $Z$, and write the output to $Y$. These functions 
 * require atleast temporary storage obtained using `gekmmSizes`. If Z and Y points 
 * to the same memory location then both temp1 and temp2 must be passed as valid 
 * memory pointers. Otherwise, only temp1 needs to be a valid memory pointer and 
 * temp2 can be NULL. All pointers should point to either x86 CPU RAM if 
 * `backend` is x86 or NVIDIA GPU RAM if `backend` is CUDA.

 * Return: Write result of GeKMM to `Z`. Return `fastKronSuccess` for no error
   or the error occurred.
 */
fastKronError sgekmmStridedBatched(fastKronHandle handle, fastKronBackend backend, 
                                   uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                                   const float* const Fs[], fastKronOp opFs, uint64_t strideF[],
                                   const float* X, fastKronOp opX, uint64_t strideX,
                                   float* Z, uint64_t strideZ, float alpha, float beta,
                                   uint32_t batchCount, const float *Y, uint64_t strideY, 
                                   float* temp1, float* temp2);

fastKronError dgekmmStridedBatched(fastKronHandle handle, fastKronBackend backend,
                                   uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                                   const double* const Fs[], fastKronOp opFs, uint64_t strideF[],
                                   const double* X, fastKronOp opX, uint64_t strideX, 
                                   double* Z, uint64_t strideZ, double alpha, double beta,
                                   uint32_t batchCount, const double *Y, uint64_t strideY,
                                   double* temp1, double* temp2);

fastKronError smkmForward(fastKronHandle handle, fastKronBackend backend, 
                            uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                            const float* X, fastKronOp opX,
                            const float* const Fs[], fastKronOp opFs,
                            float* Z, float* Intermediates[]);

fastKronError dmkmForward(fastKronHandle handle, fastKronBackend backend,
                            uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                            const double* X, fastKronOp opX,
                            const double* const Fs[], fastKronOp opFs,
                            double* Z, double* Intermediates[]);

fastKronError skmmForward(fastKronHandle handle, fastKronBackend backend, 
                            uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                            const float* const Fs[], fastKronOp opFs,
                            const float* X, fastKronOp opX,
                            float* Z, float* Intermediates[]);

fastKronError dkmmForward(fastKronHandle handle, fastKronBackend backend, 
                            uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                            const double* const Fs[], fastKronOp opFs,
                            const double* X, fastKronOp opX,
                            double* Z, double* Intermediates[]);

fastKronError smkmForwardStridedBatched(fastKronHandle handle, fastKronBackend backend, 
                                          uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                                          const float* X, fastKronOp opX, uint64_t strideX,
                                          const float* const Fs[], fastKronOp opFs, uint64_t strideF[],
                                          float* Z, uint64_t strideZ, uint32_t batchCount, 
                                          float* Intermediates[], uint64_t strideIntermediates[]);

fastKronError dmkmForwardStridedBatched(fastKronHandle handle, fastKronBackend backend,
                                          uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                                          const double* X, fastKronOp opX, uint64_t strideX,
                                          const double* const Fs[], fastKronOp opFs, uint64_t strideF[], 
                                          double* Z, uint64_t strideZ, uint32_t batchCount,
                                          double* Intermediates[], uint64_t strideIntermediates[]);

fastKronError skmmForwardStridedBatched(fastKronHandle handle, fastKronBackend backend, 
                                          uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                                          const float* const Fs[], fastKronOp opFs, uint64_t strideF[],
                                          const float* X, fastKronOp opX, uint64_t strideX,
                                          float* Z, uint64_t strideZ, uint32_t batchCount,
                                          float* Intermediates[], uint64_t strideIntermediates[]);

fastKronError dkmmForwardStridedBatched(fastKronHandle handle, fastKronBackend backend,
                                          uint32_t N, uint32_t Qs[], uint32_t Ps[], uint32_t M,
                                          const double* const Fs[], fastKronOp opFs, uint64_t strideF[],
                                          const double* X, fastKronOp opX, uint64_t strideX, 
                                          double* Z, uint64_t strideZ, uint32_t batchCount,
                                          double* Intermediates[], uint64_t strideIntermediates[]);
}