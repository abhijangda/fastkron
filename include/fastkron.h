#include <stdint.h>

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
fastKronError fastKronSetOptions(fastKronHandle handle, uint32_t options);
void fastKronDestroy(fastKronHandle handle);

const char* fastKronGetErrorString(fastKronError err);

fastKronError fastKronInitCUDA(fastKronHandle handle, void *ptrToStream, int gpus = 1, int gpusInM = -1, int gpusInK = -1, int gpuLocalKrons = -1);
//TODO: Need to provide a setcudastream function
fastKronError fastKronInitHIP(fastKronHandle handle, void *ptrToStream);
fastKronError fastKronInitX86(fastKronHandle handle);

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

// fastKronError sgekmmTune(fastKronHandle handle, uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[], 
//                        fastKronOp opX, fastKronOp opFs);
// fastKronError dgekmmTune(fastKronHandle handle, uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[], 
//                        fastKronOp opX, fastKronOp opFs);
// fastKronError igekmmTune(fastKronHandle handle, uint32_t M, uint32_t N, uint32_t Ps[], uint32_t Qs[], 
//                        fastKronOp opX, fastKronOp opFs);

// fastKronError kronSGEMMOutofCore(fastKronHandle handle, const uint32_t NumKronMats, float* x, float* kronMats[], float** result,
//                                uint32_t M, uint32_t N, uint32_t K, uint32_t KronMatCols[], uint32_t KronMatRows[], cudaStream_t stream);
// fastKronError kronSGEMMOutofCoreX(fastKronHandle handle, const uint32_t NumKronMats, float* x, float* kronMats[], float** result,
//   uint32_t M, uint32_t N, uint32_t K, uint32_t KronMatCols[], uint32_t KronMatRows[], cudaStream_t stream[]);
// fastKronError kronIGEMMOutofCoreX(fastKronHandle handle, const uint32_t NumKronMats, int* x, int* kronMats[], int** result,
//   uint32_t M, uint32_t N, uint32_t K, uint32_t KronMatCols[], uint32_t KronMatRows[], cudaStream_t stream[]);

//TODO: modify such that the results are always written to the supplied result pointer 
fastKronError kronDistributedSGEMM(fastKronHandle handle, const uint32_t NumKronMats, void* x[], void* kronMats[], void* result[],
                                 uint32_t M, uint32_t N, uint32_t K, uint32_t KronMatCols[], uint32_t KronMatRows[], 
                                 void* temp1[], void* temp2[], void* stream);

fastKronError allocDistributedX(fastKronHandle handle, void* dX[], void* hX, uint32_t M, uint32_t K);
fastKronError gatherDistributedY(fastKronHandle handle, void* dY[], void* hY, uint32_t M, uint32_t K, uint32_t NumKronMats, uint32_t KronMatCols[], uint32_t KronMatRows[]);
// fastKronError allocDistributedX(fastKronHandle handle, int* dX[], int* hX, uint32_t M, uint32_t K);
// fastKronError gatherDistributedY(fastKronHandle handle, int* dY[], int* hY, uint32_t M, uint32_t K, uint32_t NumKronMats, uint32_t KronMatCols[], uint32_t KronMatRows[]);
// fastKronError allocDistributedX(fastKronHandle handle, double* dX[], double* hX, uint32_t M, uint32_t K);
// fastKronError gatherDistributedY(fastKronHandle handle, double* dY[], double* hY, uint32_t M, uint32_t K, uint32_t NumKronMats, uint32_t KronMatCols[], uint32_t KronMatRows[]);
}