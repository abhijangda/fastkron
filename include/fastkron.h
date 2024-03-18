#pragma once

#define FastKronCHECK(cmd) do {                        \
  fastKronError e = cmd;                              \
  if(e != fastKronSuccess) {      \
    printf("Failed: FastKron error %s:%d '%s'\n",       \
        __FILE__,__LINE__,fastKronGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)                                          \

enum fastKronOp {
  fastKronOp_N = 1,
  fastKronOp_T = 2
};

enum fastKronBackend {
  fastKronBackend_NONE = 0,
  fastKronBackend_X86 = 1,
  fastKronBackend_ARM = 2,
  fastKronBackend_CUDA = 3,
  fastKronBackend_HIP = 4
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
typedef struct FastKronHandle* fastKronHandle;

fastKronError fastKronInit(fastKronHandle* handle, fastKronBackend backend);
void fastKronDestroy(fastKronHandle handle);

const char* fastKronGetErrorString(fastKronError err);

fastKronError fastKronInitCUDA(fastKronHandle handle, void *ptrToStream, int gpus = 1, int gpusInM = -1, int gpusInK = -1, int gpuLocalKrons = -1);
fastKronError fastKronInitHIP(fastKronHandle handle, void *ptrToStream);
fastKronError fastKronInitX86(fastKronHandle handlePtr);

//TODO: A different function for setting stream of handle 
fastKronError gekmmSizes(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[],
                       size_t* resultSize, size_t* tempSize);

fastKronError sgekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], float* X, 
                   fastKronOp opX, float* Fs[], fastKronOp opFs, float* Y,
                   float alpha, float beta, float *Z, float* temp1, float* temp2);
fastKronError igekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], int* X, 
                   fastKronOp opX, int* Fs[], fastKronOp opFs, int* Y,
                   int alpha, int beta, int *Z, int* temp1, int* temp2);
fastKronError dgekmm(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], double* X, 
                   fastKronOp opX, double* Fs[], fastKronOp opFs, double* Y,
                   double alpha, double beta, double *Z, double* temp1, double* temp2);

fastKronError sgekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
                       fastKronOp opX, fastKronOp opFs);
fastKronError dgekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
                       fastKronOp opX, fastKronOp opFs);
fastKronError igekmmTune(fastKronHandle handle, uint M, uint N, uint Ps[], uint Qs[], 
                       fastKronOp opX, fastKronOp opFs);

// fastKronError kronSGEMMOutofCore(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
//                                uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);
// fastKronError kronSGEMMOutofCoreX(fastKronHandle handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
//   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]);
// fastKronError kronIGEMMOutofCoreX(fastKronHandle handle, const uint NumKronMats, int* x, int* kronMats[], int** result,
//   uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]);

//TODO: modify such that the results are always written to the supplied result pointer 
fastKronError kronDistributedSGEMM(fastKronHandle handle, const uint NumKronMats, float* x[], float* kronMats[], float* result[],
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], 
                                 float* temp1[], float* temp2[], void* stream);

fastKronError allocDistributedX(fastKronHandle handle, float* dX[], float* hX, uint M, uint K);
fastKronError gatherDistributedY(fastKronHandle handle, float* dY[], float* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]);
// fastKronError allocDistributedX(fastKronHandle handle, int* dX[], int* hX, uint M, uint K);
// fastKronError gatherDistributedY(fastKronHandle handle, int* dY[], int* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]);
// fastKronError allocDistributedX(fastKronHandle handle, double* dX[], double* hX, uint M, uint K);
// fastKronError gatherDistributedY(fastKronHandle handle, double* dY[], double* hY, uint M, uint K, uint NumKronMats, uint KronMatCols[], uint KronMatRows[]);
}