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

enum fastKronOp {
  fastKronOp_N = 1,
  fastKronOp_T = 2
};

enum fastKronBackend {
  fastKronBackend_NONE = 1 << 0,
  fastKronBackend_X86 = 1 << 1,
  fastKronBackend_ARM = 1 << 2,
  fastKronBackend_CUDA = 1 << 3,
  fastKronBackend_HIP = 1 << 4
};

enum fastKronOptions {
  fastKronOptionsNone = 1 << 0,
  fastKronOptionsUseFusion = 1 << 1,
  fastKronOptionsTune = 1 << 2,
};

enum fastKronError {
  fastKronSuccess = 0,
  //FastKron not compiled with requested backend
  fastKronBackendNotAvailable = 1,
  //Invalid memory access occurred
  fastKronInvalidMemoryAccess = 2,
  //Kernel not found for requested case
  fastKronKernelNotFound = 3,
  //An argument to the API function is invalid
  fastKronInvalidArgument = 4,
  
  fastKronInvalidKMMProblem = 5, 
  //Undefined Error
  fastKronOtherError = 6,
};

extern "C" {
typedef void* fastKronHandle;

//backends is a bitwise OR
fastKronError fastKronInit(fastKronHandle* handle, uint32_t backends);
fastKronError fastKronInitAllBackends(fastKronHandle* handle);
fastKronError fastKronSetOptions(fastKronHandle handle, uint32_t options);
void fastKronDestroy(fastKronHandle handle);

uint32_t fastKronGetBackends();
const char* fastKronGetErrorString(fastKronError err);

fastKronError fastKronInitCUDA(fastKronHandle handle, void *ptrToStream);
//TODO: Need to provide a setcudastream function
fastKronError fastKronInitHIP(fastKronHandle handle, void *ptrToStream);
fastKronError fastKronInitX86(fastKronHandle handle);
fastKronError fastKronSetStream(fastKronHandle handle, fastKronBackend backend, void* ptrToStream);

//TODO: A different function for setting stream of handle
fastKronError gekmmSizes(fastKronHandle handle, uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                       size_t* yElems, size_t* tmpElems);

fastKronError sgekmm(fastKronHandle handle, fastKronBackend backend, 
                     uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                     const float* X, fastKronOp opX,
                     const float* Fs[], fastKronOp opFs,
                     float* Z, float alpha, float beta,
                     const float *Y, float* temp1, float* temp2);
fastKronError igekmm(fastKronHandle handle, fastKronBackend backend,
                     uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                     const int* X, fastKronOp opX,
                     const int* Fs[], fastKronOp opFs, 
                     int* Z, int alpha, int beta,
                     const int *Y, int* temp1, int* temp2);
fastKronError dgekmm(fastKronHandle handle, fastKronBackend backend,
                     uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[],
                     const double* X, fastKronOp opX,
                     const double* Fs[], fastKronOp opFs,
                     double* Z, double alpha, double beta,
                     const double *Y, double* temp1, double* temp2);
}